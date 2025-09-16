from torch.utils.data import Dataset
import numpy as np
import scipy.io as scio
import torch
from utils import normalize
from tqdm import tqdm
# from mydatasets.datasets import WDCMDataset as minzhichao

"""
    GT HSI: DC_HRHSI.mat shape=1280x296x191
    
    Input MSI: DC_MSI_Band10.mat shape=1280x296x10
    Input 4 x downsample HSI: DC_LRHSI_ratio4.mat
    Input 8 x downsample HSI: DC_LRHSI_ratio8.mat 160, 37, 191
    43 x 8
"""


class WDCM_dataset(Dataset):
    def __init__(self, mat_save_path, patch_size_h=16*1, patch_size_w=16*1, stride_w=3*1, stride_h=3*1, kind='train', scale_factor=8):
        super(WDCM_dataset, self).__init__()
        self.rows = 144 * 1
        self.cols = 37 * 1
        self.stride_h = stride_h
        self.stride_w = stride_w
        if scale_factor == 4:
            patch_size_h = patch_size_h * 2
            patch_size_w = patch_size_w * 2
            self.stride_h = stride_h * 2
            self.stride_w = stride_w * 2
            self.cols = self.cols * 2
            self.rows = self.rows * 2

        self.mat_save_path = mat_save_path
        self.kind = kind
        self.scale_factor = scale_factor
        # Generate samples and labels
        if self.kind == 'train':
           self.hsi_data, self.msi_data, self.label = self.generate_patches(patch_size_w, patch_size_h, is_train=True)
        if self.kind == 'validate':
            self.hsi_data, self.msi_data, self.label = self.generate_patches(patch_size_w=patch_size_w, patch_size_h=patch_size_h, is_train=False)
        if self.kind == 'test':
            self.hsi_data, self.msi_data, self.label = self.generate_patches(patch_size_w=patch_size_w, patch_size_h=patch_size_h, is_train=False)

    """
        WDCM 
        label mat: 1280x296x191
        8x down HSI: 160 x 37 x 191
        4x down HSI: (320, 74, 191)
    """
    def generate_patches(self, patch_size_w, patch_size_h, is_train):
        """
        返回训练，验证，测试的切片
        """
        ratio = self.scale_factor

        hrhsi, lrhsi, hrmsi = self.getData(self.scale_factor)
        if self.kind == 'train':
            hrhsi = hrhsi[:self.rows * ratio, :, :]
            lrhsi = lrhsi[:self.rows, :, :]
            hrhsi = hrhsi[:self.rows * ratio, :, :]
        elif self.kind == 'validate':
            hrhsi = hrhsi[self.rows * ratio:, :patch_size_h * ratio, :]
            lrhsi = lrhsi[self.rows:, :patch_size_h, :]
            hrmsi = hrmsi[self.rows * ratio:, :patch_size_h * ratio, :]
        else:
            hrhsi = hrhsi[self.rows * ratio:, patch_size_w * ratio:patch_size_w * ratio * 2, :]
            lrhsi = lrhsi[self.rows:, patch_size_w:patch_size_w * 2, :]
            hrmsi = hrmsi[self.rows * ratio:, patch_size_w * ratio:patch_size_w * ratio * 2, :]

        mat_w = lrhsi.shape[0]
        mat_h = lrhsi.shape[1]

        if not is_train:
            patch_size_w = mat_w
            patch_size_h = mat_h
            self.stride_w = patch_size_w
            self.stride_h = patch_size_h

        patches_w = (mat_w - patch_size_w) // self.stride_w + 1
        patches_h = (mat_h - patch_size_h) // self.stride_h + 1

        label_patch = np.zeros((patches_h * patches_w, patch_size_w * ratio, patch_size_h * ratio, 191), dtype=np.float32)
        hrmsi_patch = np.zeros((patches_h * patches_w, patch_size_w * ratio, patch_size_h * ratio, 10), dtype=np.float32)
        lrhsi_patch = np.zeros((patches_h * patches_w, patch_size_w, patch_size_h, 191), dtype=np.float32)
        count = 0

        print('loading and patching mat data')
        # mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
        # hrhsi = mat['hrhsi']
        # lrhsi = mat['lrhsi']
        # hrmsi = mat['hrmsi']

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, mat_w - patch_size_w + 1, self.stride_w):
            for y in range(0, mat_h - patch_size_h + 1, self.stride_h):
                label_patch[count] = hrhsi[x * ratio:(x + patch_size_w) * ratio, y * ratio:(y + patch_size_h) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size_w) * ratio, y * ratio:(y + patch_size_h) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_size_w, y:y + patch_size_h, :]
                count += 1
        print('loading completed')
        return lrhsi_patch, hrmsi_patch, label_patch

    def getData(self, ratio):
        hrhsi = scio.loadmat(self.mat_save_path + 'DC_HRHSI.mat')['S']
        lrhsi = scio.loadmat(self.mat_save_path + 'DC_LRHSI_ratio{}.mat'.format(ratio))['HSI']
        hrmsi = scio.loadmat(self.mat_save_path + 'DC_MSI_Band10.mat')['MSI']
        # Data normalization and scaling[0, 1]
        hrhsi = normalize(hrhsi)
        lrhsi = normalize(lrhsi)
        hrmsi = normalize(hrmsi)

        return hrhsi, lrhsi, hrmsi


    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2, 0, 1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]


# def main():
#     mat_save_path = '../../../datasets/WDCM/'
#     kind = 'train'
#     ratio = 8
#     mjy = WDCM_dataset(mat_save_path, kind=kind, scale_factor=ratio)
#     zhichao = minzhichao(mat_save_path, type=kind, ratio=ratio)
#     len_mine = mjy.__len__()
#     len_min = zhichao.__len__()
#     print(len_mine)
#     print(len_min)
#     for i in tqdm(range(len_mine)):
#         a = mjy.__getitem__(i)
#         b = zhichao.__getitem__(i)
#         if torch.equal(a['lrhsi'], b['lrhsi']) and torch.equal(a['hrmsi'], b['hrmsi']) and torch.equal(a['hrhsi'],
#                                                                                                        b['hrhsi']):
#             pass
#         else:
#             raise Exception('unconsist error')
#     print('consist')


# if __name__ == "__main__":
#     main()










