import os
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from collections import OrderedDict
import albumentations
import skimage.io
from scipy.special import softmax

class crossValInx(object):
    def __init__(self, csv_file):
        self.crossVal_csv = pd.read_csv(csv_file)

    def __call__(self, fold = 0):
        val_idx = self.crossVal_csv.index[self.crossVal_csv['split'] == fold].tolist()
        train_idx = list(set([x for x in range(len(self.crossVal_csv))]) - set(val_idx))
        return train_idx, val_idx

class crossValDataloader(object):
    def __init__(self, csv_file, dataset, bs = 4):
        self.inx = crossValInx(csv_file)
        self.dataset = dataset
        self.bs = bs

    def __call__(self, fold = 0):
        train_idx, val_idx = self.inx(fold)
        train = torch.utils.data.Subset(self.dataset, train_idx)
        val = torch.utils.data.Subset(self.dataset, val_idx)
        train_sampler = DistributedSampler(train)
        val_sampler = SequentialSampler(val)
        trainloader = torch.utils.data.DataLoader(train, batch_size=self.bs, shuffle=False, num_workers=4,
                                                  sampler=train_sampler,collate_fn=None, pin_memory=True,
                                                  drop_last=True)
        valloader = torch.utils.data.DataLoader(val, batch_size=self.bs, shuffle=False, num_workers=4,
                                                collate_fn=None, pin_memory=True, sampler=val_sampler,
                                                drop_last=True)
        return trainloader, valloader, train_sampler

class PandaPatchDataset(Dataset):
    """
    gls2isu = {"0+0":0,'negative':0,'3+3':1,'3+4':2,'4+3':3,'4+4':4,'3+5':4,'5+3':4,'4+5':5,'5+4':5,'5+5':5}
    """
    gls = {"0+0": [0, 0], 'negative': [0, 0], '3+3': [1, 1], '3+4': [1, 2], '4+3': [2, 1], '4+4': [2, 2],
           '3+5': [1, 3], '5+3': [3, 1], '4+5': [2, 3], '5+4': [3, 2], '5+5': [3, 3]}
    """Panda Tile dataset. With fixed tiles for each slide."""
    def __init__(self, csv_file, image_dir, image_size, N = 36, transform=None, rand=False, mode="br"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            N (interger): Number of tiles selected for each slide.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_csv = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        self.N = N
        self.rand = rand
        self.mode = mode

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        result = OrderedDict()
        img_id = self.train_csv.loc[idx, 'image_id']
        name = self.train_csv.image_id[idx]
        if self.mode == "br":
            tile_pix = str(self.train_csv.tile_blueratio[idx])
        else:
            tile_pix = str(self.train_csv.tile_pixel[idx])

        tile_pix = np.asarray(tile_pix.split(",")[:-1]).astype(int)
        idxes = idx_selection(tile_pix, self.N, "deterministic")
        fnames = [os.path.join(self.image_dir, img_id + '_' + str(i) + '.png')
                                for i in idxes]

        imgs = []
        for i, fname in enumerate(fnames):
            img = self.open_image(fname)
            imgs.append({'img': img, 'idx': i})

        if self.rand: ## random shuffle the order of tiles
            idxes = np.random.choice(list(range(self.N)), self.N, replace=False)
        else:
            idxes = list(range(self.N))

        n_row_tiles = int(np.sqrt(self.N))

        images = np.zeros((self.image_size * n_row_tiles, self.image_size * n_row_tiles, 3)).astype(np.uint8)
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
                if len(imgs) > idxes[i]:
                    this_img = imgs[idxes[i]]['img'].astype(np.uint8)
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                # this_img = 255 - this_img ## todo: see how this trik plays out
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * self.image_size
                w1 = w * self.image_size
                images[h1:h1 + self.image_size, w1:w1 + self.image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255.0
        mean = np.asarray([0.79667089, 0.59347025, 0.75775308])
        std = np.asarray([0.07021654, 0.13918451, 0.08442586])
        images = (images - mean)/(std) ## normalize the image
        images = images.transpose(2, 0, 1)
        label = np.zeros(5).astype(np.float32)
        isup_grade = self.train_csv.loc[idx, 'isup_grade']
        gleason_score = self.gls[self.train_csv.loc[idx, 'gleason_score']]
        primary_gls = np.zeros(4).astype(np.float32)
        secondary_gls = np.zeros(4).astype(np.float32)
        datacenter = self.train_csv.loc[idx, 'data_provider']
        label[:isup_grade] = 1.
        result['img'] = torch.tensor(images)
        result['isup_grade'] = torch.tensor(label)
        result['datacenter'] = datacenter
        primary_gls[:gleason_score[0]] = 1.
        secondary_gls[:gleason_score[1]] = 1.
        result['primary_gls'] = torch.tensor(primary_gls)
        result['secondary_gls'] = torch.tensor(secondary_gls)
        result['name'] = name
        return result

    def open_image(self, fn, convert_mode='RGB', after_open=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            x = Image.open(fn).convert(convert_mode)
            x= np.asarray(x)
        if after_open:
            x = after_open(x)
        return x

def idx_selection(logit_list, N = 36, mode = "deterministic"):
    """

    :param logit_list: numpy array, the larger, the more probability to select
    :param mode:
    :return:
    """
    if mode == "deterministic":
        if len(logit_list) >= N:
            idx = list(np.argsort(logit_list)[::-1][:N])
        else:
            idx = list(range(len(logit_list)))
            idx += list(np.argsort(logit_list)[::-1][:N - len(logit_list)])
    else: ## random mode
        logit_list = (logit_list - np.min(logit_list)) / (np.max(logit_list) - np.min(logit_list) + 1e-12)
        prob = softmax(logit_list)
        idx = np.random.choice(len(logit_list), N, p=prob, replace=True)
    return idx

def data_transform():
    tsfm = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        # albumentations.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        albumentations.HueSaturationValue(hue_shift_limit=7, sat_shift_limit=20)
        # albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,)
    ])
    return tsfm

def dataloader_collte_fn(batch):
    result = OrderedDict()
    imgs = [item['img'] for item in batch]
    imgs = torch.stack(imgs)
    target = [item['isup_grade'] for item in batch]
    target = torch.stack(target)
    datacenter = [item['datacenter'] for item in batch]
    primary_gls = [item['primary_gls'] for item in batch]
    secondary_gls = [item['secondary_gls'] for item in batch]
    name = [item['name'] for item in batch]
    result['img'] = imgs
    result['isup_grade'] = target
    result['datacenter'] = datacenter
    result['primary_gls'] = primary_gls
    result['secondary_gls'] = secondary_gls
    result['name'] = name
    return result


if __name__ == "__main__":
    ## input files and folders
    nfolds = 4
    bs = 4
    sz = 256
    csv_file = './csv_pkl_files/{}_fold_train.csv'.format(nfolds)
    image_dir = './panda-32x256x256-tiles-data/train/'
    ## image transformation
    tsfm = data_transform()
    ## dataset, can fetch data by dataset[idx]
    dataset = PandaPatchDataset(csv_file, image_dir, sz, transform=tsfm, N=16, rand=True)
    ## dataloader
    dataloader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=4, collate_fn=dataloader_collte_fn)

    ## fetch data from dataloader
    data = iter(dataloader).next()
    print("image size:{}, target sise:{}.".format(data['img'].size(), data['isup_grade'].size()))

    ## cross val dataloader
    crossValData = crossValDataloader(csv_file, dataset, bs)
    trainloader, valloader = crossValData(0)
    data = iter(valloader).next()
    print("image size:{}, target sise:{}.".format(data['img'].size(), data['isup_grade'].size()))