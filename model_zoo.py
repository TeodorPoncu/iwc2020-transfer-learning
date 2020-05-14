import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision as tv


class Self_Attn_FM(nn.Module):
    """ Self attention Layer for Feature Map dimension"""

    def __init__(self, in_dim, latent_dim=8):
        super(Self_Attn_FM, self).__init__()
        self.channel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // latent_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // latent_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key = self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x N x N, N = H x W
        energy = torch.bmm(proj_query, proj_key)
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)

        out = self.gamma * out + x
        return out, attention

class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, relative):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative

        init_gain = 2 ** 0.5

        param_std = init_gain * (self.in_channels * (self.out_channels - self.dv) * self.kernel_size ** 2) ** (-0.5)
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, padding=1)
        self.conv_out.weight = torch.nn.Parameter(torch.randn_like(self.conv_out.weight) * param_std)

        param_std = init_gain * (self.in_channels * (2 * self.dk + self.dv)) ** (-0.5)
        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.qkv_conv.weight = torch.nn.Parameter(torch.randn_like(self.qkv_conv.weight) * param_std)

        param_std = init_gain * (self.dv * self.dv) ** (-0.5)
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)
        self.attn_out.weight = torch.nn.Parameter(torch.randn_like(self.attn_out.weight) * param_std)

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(self.conv_out.weight.device)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(self.conv_out.weight.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def compute_flat_qkv(self, x, dk, dv, Nh):
        N, _, H, W = x.size()
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        key_rel_w = nn.Parameter(torch.randn((2 * W - 1, dk), requires_grad=True)).to(self.conv_out.weight.device)
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, "w")

        key_rel_h = nn.Parameter(torch.randn((2 * H - 1, dk), requires_grad=True)).to(self.conv_out.weight.device)
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

class BaseLine(nn.Module):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        self.model = tv.models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(in_features=1024, out_features=NUM_CLASSES)

    def forward(self, x):
        x = self.model(x)
        return x

    def get_backbone_parameters(self):
        return list(self.model.features.parameters())

    def get_additional_parameters(self):
        params = list(self.model.classifier.parameters())
        return params

class AAConvModelNoRes(nn.Module):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        self.backbone = tv.models.densenet121(pretrained=True).features
        self.attn_block = AugmentedConv(1024, 512, kernel_size=3, dk=64, dv=64, Nh=8, relative=True)
        self.norm_block = nn.BatchNorm2d(512)
        self.sqsh_block = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.backbone(x))
        x = self.attn_block(x)
        x = self.norm_block(F.relu(x))
        x = self.sqsh_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_backbone_parameters(self):
        return list(self.backbone.parameters())

    def get_additional_parameters(self):
        params = list(self.attn_block.parameters()) + list(self.norm_block.parameters()) + \
                 list(self.sqsh_block.parameters()) + list(self.classifier.parameters())
        return params

class NonLocalModel(nn.Module):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        self.backbone = tv.models.densenet121(pretrained=True).features
        self.attn_block = Self_Attn_FM(1024)
        self.proj_block = nn.Conv2d(1024, 512, kernel_size=1)
        self.resc_block = nn.Conv2d(1024, 512, kernel_size=1)
        self.norm_block = nn.BatchNorm2d(512)
        self.sqsh_block = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)

    def forward(self, x):
        res = self.backbone(x)
        x, _ = self.attn_block(res)
        x = F.relu(x)
        x = (self.proj_block(x) + self.resc_block(res)) * (2 ** -0.5)
        x = self.norm_block(F.relu(x))
        x = self.sqsh_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_backbone_parameters(self):
        return list(self.backbone.parameters())

    def get_additional_parameters(self):
        params = list(self.attn_block.parameters()) + list(self.resc_block.parameters()) + list(self.norm_block.parameters()) + \
                 list(self.sqsh_block.parameters()) + list(self.classifier.parameters()) + list(self.proj_block.parameters())
        return params

class NonLocalModelNoRes(nn.Module):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        self.backbone = tv.models.densenet121(pretrained=True).features
        self.attn_block = Self_Attn_FM(1024)
        self.proj_block = nn.Conv2d(1024, 512, kernel_size=1)
        self.norm_block = nn.BatchNorm2d(512)
        self.sqsh_block = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)

    def forward(self, x):
        res = self.backbone(x)
        x, _ = self.attn_block(res)
        x = self.norm_block(F.relu(self.proj_block(x)))
        x = self.sqsh_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_backbone_parameters(self):
        return list(self.backbone.parameters())

    def get_additional_parameters(self):
        params = list(self.attn_block.parameters()) + list(self.norm_block.parameters()) + \
                 list(self.sqsh_block.parameters()) + list(self.classifier.parameters()) + list(self.proj_block.parameters())
        return params

class AttrousConvModel(nn.Module):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        self.backbone = tv.models.densenet121(pretrained=True).features
        self.attrous = nn.ModuleList([
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, dilation=3, padding=3),
        ])
        self.norm_block = nn.BatchNorm2d(512)
        self.sqsh_block = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.proj_block = nn.Conv2d(1024 * 3, 512, kernel_size=1)
        self.resc_block = nn.Conv2d(1024, 512, kernel_size=1)
        self.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)

    def forward(self, x):
        res = self.backbone(x)
        x = torch.cat([attrous(res) for attrous in self.attrous], dim=1)
        x = self.norm_block(F.relu((self.proj_block(x) + self.resc_block(res)) * (2 ** -0.5)))
        x = self.sqsh_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AttrousConvModelNoRes(nn.Module):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        self.backbone = tv.models.densenet121(pretrained=True).features
        self.attrous = nn.ModuleList([
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, dilation=3, padding=3),
        ])
        self.norm_block = nn.BatchNorm2d(512)
        self.sqsh_block = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.proj_block = nn.Conv2d(1024 * 3, 512, kernel_size=1)
        self.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.cat([attrous(x) for attrous in self.attrous], dim=1)
        x = self.norm_block(F.relu(self.proj_block(x)))
        x = self.sqsh_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_backbone_parameters(self):
        return list(self.backbone.parameters())

    def get_additional_parameters(self):
        params = list(self.attrous.parameters()) + list(self.norm_block.parameters()) + list(self.resc_block.parameters()) + \
                 list(self.sqsh_block.parameters()) + list(self.classifier.parameters()) + list(self.proj_block.parameters())
        return params

class AAConvModel(nn.Module):
    def __init__(self, NUM_CLASSES):
        super().__init__()
        init_gain = 2 ** 0.5
        self.backbone = tv.models.densenet121(pretrained=True).features
        self.attn_block = AugmentedConv(1024, 512, kernel_size=3, dk=64, dv=64, Nh=8, relative=True)
        self.resc_block = nn.Conv2d(1024, 512, kernel_size=1)
        self.resc_block.weight = torch.nn.Parameter(torch.randn_like(self.resc_block.weight) * ((1024 * 512) ** -0.5) * init_gain)
        self.norm_block = nn.BatchNorm2d(512)
        self.sqsh_block = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512, out_features=NUM_CLASSES)
        self.classifier.weight = torch.nn.Parameter(torch.randn_like(self.classifier.weight) * (512 ** -0.5) * init_gain)

    def forward(self, x):
        res = self.backbone(x)
        x = F.relu(res)
        x = (self.attn_block(x) + self.resc_block(res)) * (2 ** -0.5)
        x = self.norm_block(F.relu(x))
        x = self.sqsh_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_backbone_parameters(self):
        return list(self.backbone.parameters())

    def get_additional_parameters(self):
        params = list(self.attn_block.parameters()) + list(self.resc_block.parameters()) + list(self.norm_block.parameters()) + \
                 list(self.sqsh_block.parameters()) + list(self.classifier.parameters())
        return params

