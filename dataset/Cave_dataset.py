import os.path

import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
from utils import augment


class CaveDataset(Dataset):
    def __init__(self, mat_save_path, patch_size=16*1, stride=6*1, kind='train', aug=False):
        super(CaveDataset, self).__init__()
        tmp = scio.loadmat(os.path.join(mat_save_path, '1.mat'))
        h, w, _ = tmp['lrhsi'].shape
        H, W, _ = tmp['hrmsi'].shape
        self.r = H // h
        if self.r == 4:
            patch_size = patch_size * 2
            stride = stride * 2
        self.aug = aug
        self.mat_save_path = mat_save_path
        self.stride = stride
        self.patch_size = patch_size
        if kind != 'train':
            self.patch_size = h
            self.stride = self.patch_size
        self.patch_per_line = (w - self.patch_size) // stride + 1
        self.patch_per_colum = (h - self.patch_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        mat_start = 1
        mat_end = 22
        if kind == 'validate':
            mat_start = 23
            mat_end = 27
        elif kind == 'test':
            mat_start = 28
            mat_end = 32
        self.mats = self.load_mat(mat_start, mat_end)
        self.length = len(self.mats) * self.patch_per_img

    def __getitem__(self, index):
        stride = self.stride
        patch_size = self.patch_size
        img_idx, patch_idx = index // self.patch_per_img, index % self.patch_per_img
        h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
        mat = self.mats[img_idx]
        hrhsi = mat['hrhsi']
        hrmsi = mat['hrmsi']
        lrhsi = mat['lrhsi']

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)

        r = self.r
        lrhsi = lrhsi[h_idx * stride:h_idx * stride + patch_size, w_idx * stride:w_idx * stride + patch_size, :]
        hrmsi = hrmsi[h_idx * stride * r:(h_idx * stride + patch_size) * r, w_idx * stride * r:(w_idx * stride + patch_size) * r, :]
        hrhsi = hrhsi[h_idx * stride * r:(h_idx * stride + patch_size) * r, w_idx * stride * r:(w_idx * stride + patch_size) * r, :]

        #  data augment
        if self.aug:
            rotTimes = random.randint(1, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hrhsi = augment(img=hrhsi, rotTimes=rotTimes, vFlip=vFlip, hFlip=hFlip)
            hrmsi = augment(img=hrmsi, rotTimes=rotTimes, vFlip=vFlip, hFlip=hFlip)
            lrhsi = augment(img=lrhsi, rotTimes=rotTimes, vFlip=vFlip, hFlip=hFlip)

        hrhsi = np.transpose(hrhsi, (2, 0, 1))
        hrmsi = np.transpose(hrmsi, (2, 0, 1))
        lrhsi = np.transpose(lrhsi, (2, 0, 1))

        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def load_mat(self, start, end):
        mats = []
        for i in range(start, end + 1):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            mats.append(mat)
        return mats

    def __len__(self):
        return self.length









