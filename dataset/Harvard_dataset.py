import random

from torch.utils.data import Dataset
import os
import numpy as np
import scipy.io as scio
import torch
from tqdm import tqdm
from utils import augment


class HarvardDataset(Dataset):
    def __init__(self, mat_save_path, patch_size=16, stride=10, kind='train', aug=False):
        super(HarvardDataset, self).__init__()
        """
            label: 1040 x 1392 x 31
            8x hsi: 130 x 174 x 31
        """
        tmp = scio.loadmat(os.path.join(mat_save_path, '1.mat'))
        h, w, _ = tmp['lrhsi'].shape
        H, W, _ = tmp['hrmsi'].shape
        self.r = H // h
        if self.r == 4:
            patch_size = patch_size * 2
            stride = stride * 2
        self.aug = aug
        self.mat_save_path = mat_save_path
        self.patch_size = patch_size
        self.stride = stride
        self.kind = kind
        self.patch_per_line = (w - patch_size) // stride + 1
        self.patch_per_colum = (h - patch_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        # training:
        mat_start = 1
        mat_end = 34

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

    def __len__(self):
        return self.length

    def load_mat(self, start, end):
        mats = []
        for i in range(start, end + 1):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            mats.append(mat)
        return mats


class Val_test_dataset(Dataset):
    def __init__(self, mat_save_path, kind='validate'):
        super(Val_test_dataset, self).__init__()
        self.mat_save_path = mat_save_path
        mat_start = 43
        mat_end = 50
        if kind == 'test':
            mat_start = 35
            mat_end = 42
        self.mats = self.load_mat(mat_start, mat_end)
        self.length = len(self.mats)

    def load_mat(self, start, end):
        mats = []
        for i in range(start, end + 1):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            mats.append(mat)
        return mats

    def __getitem__(self, index):
        mat = self.mats[index]
        r = mat['hrhsi'].shape[0] // mat['lrhsi'].shape[0]
        hrhsi = mat['hrhsi'][:1024, :1024, :]
        hrmsi = mat['hrmsi'][:1024, :1024, :]
        lrhsi = mat['lrhsi'][:256, :256, :] if r == 4 else mat['lrhsi'][:128, :128, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)

        hrhsi = np.transpose(hrhsi, (2, 0, 1))
        hrmsi = np.transpose(hrmsi, (2, 0, 1))
        lrhsi = np.transpose(lrhsi, (2, 0, 1))

        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.length

class Val_test_dataset2(Dataset):
    def __init__(self, mat_save_path, patch_size=64, stride=64, kind='validate', aug=False):
        super(Val_test_dataset2, self).__init__()
        """
            label: 1040 x 1392 x 31
            8x hsi: 130 x 174 x 31
        """
        tmp = scio.loadmat(os.path.join(mat_save_path, '1.mat'))
        h, w, _ = tmp['lrhsi'].shape
        H, W, _ = tmp['hrmsi'].shape
        self.r = H // h
        if self.r == 4:
            patch_size = patch_size * 2
            stride = stride * 2
        self.aug = aug
        self.mat_save_path = mat_save_path
        self.patch_size = patch_size
        self.stride = stride
        self.kind = kind
        self.patch_per_line = (w - patch_size) // stride + 1
        self.patch_per_colum = (h - patch_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        # validate:
        mat_start = 43
        mat_end = 50
        if kind == 'test':
            mat_start = 35
            mat_end = 42

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

    def __len__(self):
        return self.length

    def load_mat(self, start, end):
        mats = []
        for i in range(start, end + 1):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            mats.append(mat)
        return mats

# def main():
#     mat_save_path = '../../../datasets/Harvard_mat/4/'
#     kind = 'train'
#     harvard1 = H1(mat_save_path, kind=kind)
#     # harvard2 = HarvardDataset(mat_save_path, kind=kind)
#     harvard2 = Val_test_dataset(mat_save_path, kind=kind)
#     len_1 = harvard1.__len__()
#     len_2 = harvard2.__len__()
#     print(f"len_cave1: {len_1}, len_cave2: {len_2}")
#     assert len_1 == len_2
#     for i in tqdm(range(len_2)):
#         a = harvard1.__getitem__(i)
#         b = harvard2.__getitem__(i)
#         if torch.equal(a['lrhsi'], b['lrhsi']) and torch.equal(a['hrmsi'], b['hrmsi']) and torch.equal(a['hrhsi'],
#                                                                                                        b['hrhsi']):
#             pass
#         else:
#             raise Exception('unconsist error')
#     print('consist')
#
#
# if __name__ == '__main__':
#     main()












