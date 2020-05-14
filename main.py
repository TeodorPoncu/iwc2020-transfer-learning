import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision.transforms.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import PIL.Image as Image
from PIL import ImageFile
import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_zoo import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_W, IMG_H = (512, 512)

ID_COLNAME = 'file_name'
ANSWER_COLNAME = 'category_id'
TRAIN_IMGS_DIR = 'train'
TEST_IMGS_DIR = 'test'

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

train_augmentation = transforms.Compose([
    transforms.Resize((IMG_W, IMG_H)),
    transforms.ToTensor(),
    normalizer,
])

val_augmentation = transforms.Compose([
    transforms.Resize((IMG_W, IMG_H)),
    transforms.ToTensor(),
    normalizer,
])


with open('iwildcam2020_train_annotations.json') as json_file:
    train_data = json.load(json_file)

df_train = pd.DataFrame({'id': [item['id'] for item in train_data['annotations']],
                         'category_id': [item['category_id'] for item in train_data['annotations']],
                         'image_id': [item['image_id'] for item in train_data['annotations']],
                         'file_name': [item['file_name'] for item in train_data['images']]})
df_train.head()

df_image = pd.DataFrame.from_records(train_data['images'])

indices = []
for _id in df_image[df_image['location'] == 537]['id'].values:
    indices.append(df_train[df_train['image_id'] == _id].index)

for the_index in indices:
    df_train = df_train.drop(df_train.index[the_index])

indices = []
dropped_imgs = 0
with open('dump_paths.txt', 'w') as f:
    for i in df_train['file_name']:
        if not os.path.exists(os.path.join('train', i)):
            dropped_imgs += 1
            df_train.drop(df_train.loc[df_train['file_name'] == i].index, inplace=True)
            f.write(os.path.join('train', i) + '\n')

print('Dropped :{} imgs'.format(dropped_imgs))

with open('iwildcam2020_test_information.json') as f:
    test_data = json.load(f)

df_test = pd.DataFrame.from_records(test_data['images'])
df_test.head()

train_df, test_df = train_test_split(df_train[[ID_COLNAME, ANSWER_COLNAME]],test_size=0.15,shuffle=True)
CLASSES_TO_USE = df_train['category_id'].unique()
NUM_CLASSES = len(CLASSES_TO_USE)
CLASSMAP = dict(
    [(i, j) for i, j
     in zip(CLASSES_TO_USE, range(NUM_CLASSES))
     ]
)
REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])



class IMetDataset(Dataset):

    def __init__(self,
                 df,
                 images_dir,
                 n_classes=NUM_CLASSES,
                 id_colname=ID_COLNAME,
                 answer_colname=ANSWER_COLNAME,
                 label_dict=CLASSMAP,
                 transforms=None
                 ):
        self.df = df
        self.images_dir = images_dir
        self.n_classes = n_classes
        self.id_colname = id_colname
        self.answer_colname = answer_colname
        self.label_dict = label_dict
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_name = img_id  # + self.img_ext
        img_path = os.path.join(self.images_dir, img_name)

        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.answer_colname is not None:
            label_F1 = torch.zeros((self.n_classes,), dtype=torch.float32)
            label_F1[self.label_dict[cur_idx_row[self.answer_colname]]] = 1.0

            #label = self.label_dict[cur_idx_row[self.answer_colname]]
            label = torch.zeros((self.n_classes,), dtype=torch.long)
            label = self.label_dict[cur_idx_row[self.answer_colname]]
            #label[self.label_dict[cur_idx_row[self.answer_colname]]] = 1.0

            return img, label, label_F1

        else:
            return img, img_id


def f1_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


def train_one_epoch(model, train_loader, criterion, optimizer, steps_upd_logging=250):
    model.train();

    total_loss = 0.0

    train_tqdm = tqdm(train_loader)

    for step, (features, targets, label_F1) in enumerate(train_tqdm):
        features, targets = features.cuda(), targets.cuda()
        for opt in optimizer:
            opt.zero_grad()

        logits = model(features)

        loss = criterion(logits, label_F1.cuda())
        loss.backward()

        for opt in optimizer:
            opt.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            logstr = f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}'
            print(logstr)
            #train_tqdm.set_description(logstr)

    return total_loss / (step + 1)


def validate(model, valid_loader, criterion, need_tqdm=False):
    model.eval();

    test_loss = 0.0
    TH_TO_ACC = 0.5

    true_ans_list = []
    preds_cat = []

    with torch.no_grad():
        valid_iterator = tqdm(valid_loader)

        for step, (features, targets, label_F1) in enumerate(valid_iterator):
            features, targets = features.cuda(), targets.cuda()

            #targets = targets.view(-1, 1)
            logits = model(features)
            loss = criterion(logits, label_F1.cuda())

            test_loss += loss.item()
            true_ans_list.append(label_F1.cuda())
            preds_cat.append(torch.sigmoid(logits))

        all_true_ans = torch.cat(true_ans_list)
        all_preds = torch.cat(preds_cat)

        f1_eval = f1_score(all_true_ans, all_preds).item()

    logstr = f'Mean val f1: {round(f1_eval, 5)}'
    return test_loss / (step + 1), f1_eval


def get_subm_answers(model, subm_dataloader, need_tqdm=False):
    model.eval()
    preds_cat = []
    ids = []

    with torch.no_grad():
        subm_iterator = tqdm(subm_dataloader)

        for step, (features, subm_ids) in enumerate(subm_iterator):
            features = features.cuda()

            logits = model(features)
            preds_cat.append(torch.sigmoid(logits))
            ids = ids + subm_ids

        all_preds = torch.cat(preds_cat)
        all_preds = torch.argmax(all_preds, dim=1).int().cpu().numpy()
    return all_preds, ids

def process_one_id(id_classes_str):
    if id_classes_str:
        return REVERSE_CLASSMAP[int(id_classes_str)]
    else:
        return id_classes_str


if __name__ == '__main__':

    devices = [torch.device('cuda:{}'.format(i)) for i in range(2)]
    model = AAConvModel(NUM_CLASSES).cuda()
    optim_backbone = torch.optim.AdamW(params=model.get_backbone_parameters(), lr=0.00001, weight_decay=0.00)
    optim_classifier = torch.optim.AdamW(params=model.get_additional_parameters(), lr=0.0005)
    model = torch.nn.DataParallel(model, device_ids=devices)

    optims = [optim_backbone, optim_classifier]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_classifier, factor=0.5, patience=3)

    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_hard = torch.nn.BCEWithLogitsLoss()

    train_dataset = IMetDataset(train_df, 'train', transforms=train_augmentation)
    test_dataset = IMetDataset(test_df, 'train', transforms=val_augmentation)

    BS = 16

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)

    TRAIN_LOGGING_EACH = 80

    train_losses = []
    valid_losses = []
    valid_f1s = []
    best_model_f1 = 0.0
    best_model = None
    best_model_ep = 0

    for epoch in range(1, 1 + 1):
        ep_logstr = f"Starting {epoch} epoch..."
        print(ep_logstr)
        tr_loss = train_one_epoch(model, train_loader, criterion, optims, steps_upd_logging=20)
        train_losses.append(tr_loss)
        tr_loss_logstr = f'Mean train loss: {round(tr_loss, 5)}'
        print(tr_loss_logstr)
        valid_loss, valid_f1 = validate(model, test_loader, criterion)
        valid_losses.append(valid_loss)
        valid_f1s.append(valid_f1)
        val_loss_logstr = f'Mean valid loss: {round(valid_loss, 5)}'
        scheduler.step(valid_loss)
        print(val_loss_logstr, 'F1 score: {}'.format(valid_f1))

        if valid_f1 >= best_model_f1:
            best_model = model
            best_model_f1 = valid_f1
            best_model_ep = epoch

    xs = list(range(1, len(train_losses) + 1))
    plt.plot(xs, train_losses, label='Train loss')
    plt.plot(xs, valid_losses, label = 'Val loss')
    plt.plot(xs, valid_f1s, label='Val f1')
    plt.legend()
    plt.xticks(xs)
    plt.xlabel('Epochs')

    SAMPLE_SUBMISSION_DF = pd.read_csv('sample_submission.csv')
    SAMPLE_SUBMISSION_DF.head()

    SAMPLE_SUBMISSION_DF.rename(columns={'Id': 'file_name', 'Category': 'category_id'}, inplace=True)
    SAMPLE_SUBMISSION_DF['file_name'] = SAMPLE_SUBMISSION_DF['file_name'] + '.jpg'
    SAMPLE_SUBMISSION_DF.head()

    SUMB_BS = 16

    subm_dataset = IMetDataset(SAMPLE_SUBMISSION_DF, TEST_IMGS_DIR, transforms=val_augmentation, answer_colname=None)
    subm_dataloader = DataLoader(subm_dataset, batch_size=SUMB_BS, num_workers=8, shuffle=False, pin_memory=True)

    subm_preds, submids = get_subm_answers(best_model, subm_dataloader, True)
    ans_dict = dict(zip(submids, subm_preds.astype(str)))
    df_to_process = (
        pd.DataFrame
            .from_dict(ans_dict, orient='index', columns=['Category'])
            .reset_index()
            .rename({'index': 'Id'}, axis=1)
    )
    df_to_process['Id'] = df_to_process['Id'].map(lambda x: str(x)[:-4])
    df_to_process.head()

    df_to_process['Category'] = df_to_process['Category'].apply(process_one_id)
    df_to_process.head()

    df_to_process.to_csv('submission.csv', index=False)









