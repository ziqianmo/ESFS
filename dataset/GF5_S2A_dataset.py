import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
import os


class GF5_S2A_dataset(Dataset):
    def __init__(self, mat_save_path='/home8/mzq/data/GF5_S2A/GF5.mat', patch_size=32, stride=12, kind='train'):
        self.test = False
        if kind == 'train':
            self.data_msi, self.data_hsi, self.data_label = self.generate_patches(mat_save_path=mat_save_path,
                                                                                  patch_size=patch_size, stride=stride)
        if kind == 'validate':
            self.data_msi, self.data_hsi, self.data_label = self.generate_val_set(mat_save_path=mat_save_path,
                                                                                  patch_size=64, num=10)
        if kind == 'test':
            self.test = True
            self.data_hsi, self.data_msi = self.generate_test_set()

    def generate_patches(self, mat_save_path, patch_size, stride):
        mat = scio.loadmat(mat_save_path)
        w_mat_msi, h_mat_msi, c_mat_msi = mat['hrmsi'].shape
        w_mat_hsi, h_mat_hsi, c_mat_hsi = mat['lrhsi'].shape
        scale_factor = w_mat_msi // w_mat_hsi
        patches_w = (w_mat_hsi-patch_size)//stride + 1  # number of patches on width dimension
        patches_h = (h_mat_hsi-patch_size)//stride + 1  # number of patches on height dimension

        # data type conversion
        hsi_mat = mat['lrhsi']
        msi_mat = mat['hrmsi']
        label_mat = mat['hrhsi']

        if hsi_mat.dtype != np.float32:
            hsi_mat = hsi_mat.astype(np.float32)
        if msi_mat.dtype != np.float32:
            msi_mat = msi_mat.astype(np.float32)
        if label_mat.dtype != np.float32:
            label_mat = label_mat.astype(np.float32)

        patches_label = np.zeros([patches_h*patches_w, patch_size*scale_factor, patch_size*scale_factor, c_mat_hsi], dtype=np.float32)
        patches_msi = np.zeros([patches_h*patches_w, patch_size*scale_factor, patch_size*scale_factor, c_mat_msi], dtype=np.float32)
        patches_hsi = np.zeros([patches_h*patches_w, patch_size, patch_size, c_mat_hsi], dtype=np.float32)
        count = 0
        print('loading and patching mat data')
        for i in tqdm(range(0, w_mat_hsi-patch_size+1, stride)):
            for j in range(0, h_mat_hsi-patch_size+1, stride):
                patches_label[count] = label_mat[i*scale_factor:(i+patch_size)*scale_factor, j*scale_factor:(j+patch_size)*scale_factor, :]
                patches_msi[count] = msi_mat[i*scale_factor:(i+patch_size)*scale_factor, j*scale_factor:(j+patch_size)*scale_factor, :]
                patches_hsi[count] = hsi_mat[i:(i+patch_size), j:(j+patch_size), :]
                count += 1

        return patches_msi, patches_hsi, patches_label

    def generate_val_set(self, mat_save_path, patch_size=128, num=10):
        np.random.seed(2411)
        mat = scio.loadmat(mat_save_path)
        w_mat_hsi, h_mat_hsi, c_mat_hsi = mat['lrhsi'].shape
        w_mat_msi, h_mat_msi, c_mat_msi = mat['hrmsi'].shape
        scale_factor = w_mat_msi // w_mat_hsi
        hsi_mat = mat['lrhsi']
        msi_mat = mat['hrmsi']
        label_mat = mat['hrhsi']
        patches_label = np.zeros([num, patch_size * scale_factor, patch_size * scale_factor, c_mat_hsi], dtype=np.float32)
        patches_msi = np.zeros([num, patch_size * scale_factor, patch_size * scale_factor, c_mat_msi],dtype=np.float32)
        patches_hsi = np.zeros([num, patch_size, patch_size, c_mat_hsi], dtype=np.float32)

        print('loading and patching mat data')
        for i in tqdm(range(num)):
            x = random.randint(patch_size, w_mat_hsi-patch_size)
            y = random.randint(patch_size, w_mat_hsi-patch_size)
            patches_label[i] = label_mat[x*scale_factor:(x+patch_size)*scale_factor, y*scale_factor:(y+patch_size)*scale_factor, :]
            patches_msi[i] = msi_mat[x*scale_factor:(x+patch_size)*scale_factor, y*scale_factor:(y+patch_size)*scale_factor, :]
            patches_hsi[i] = hsi_mat[x:x+patch_size, y:y+patch_size, :]
        return patches_msi, patches_hsi, patches_label

    def generate_test_set(self):
        """
        hrmsi: 576x576x4
        lrhsi: 192x192x280
        :return:
        """
        path = '/home8/mzq/data/GF5_S2A/real/'
        file_names = os.listdir(path)
        hsi = np.zeros((len(file_names), 192, 192, 280), dtype=np.float32)
        msi = np.zeros((len(file_names), 576, 576, 4), dtype=np.float32)
        for i, file_name in enumerate(file_names):
            file_mat = scio.loadmat(os.path.join(path, file_name))
            hsi[i] = file_mat['lrhsi']
            msi[i] = file_mat['hrmsi']

        return hsi, msi


    def __getitem__(self, index):
        msi = torch.tensor(np.transpose(self.data_msi[index], [2, 0, 1]))
        hsi = torch.tensor(np.transpose(self.data_hsi[index], [2, 0, 1]))
        if self.test:
            item = {
                    'lrhsi': hsi,
                    'hrmsi': msi}
        else :
            label = torch.tensor(np.transpose(self.data_label[index], [2, 0, 1]))
            item = {'hrhsi': label,
                    'lrhsi': hsi,
                    'hrmsi': msi}

        return item

    def __len__(self):
        return self.data_msi.shape[0]


def main():
    ds = GF5_S2A_dataset(mat_save_path='/home8/mzq/data/GF5_S2A/GF5.mat', kind='validate')
    tmp = ds.__getitem__(100)

if __name__ == '__main__':
    main()









