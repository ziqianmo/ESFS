import os.path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join
from dataset.Cave_dataset import CaveDataset
from dataset.Harvard_dataset import HarvardDataset
from dataset.Harvard_dataset import Val_test_dataset2 as Harvard_val
from dataset.GF5_S2A_dataset import GF5_S2A_dataset
from dataset.WDCM_dataset import WDCM_dataset
import argsParser
from dataset.path_config import dataset_dirs


def get_dataloader(args):
    # Set dataset directory based on the specified dataset
    dataset_dir = join(dataset_dirs[args.dataset], str(args.ratio)) + '/'
    if args.dataset == 'cave':
        data_train = CaveDataset(mat_save_path=dataset_dir, kind='train')
        data_validate = CaveDataset(mat_save_path=dataset_dir, kind='validate')
    elif args.dataset == 'harvard':
        data_train = HarvardDataset(dataset_dir, kind='train')
        data_validate = Harvard_val(dataset_dir, kind='validate')
    elif args.dataset == 'GF5_S2A':
        data_train = GF5_S2A_dataset(mat_save_path=dataset_dirs[args.dataset], kind='train')
        data_validate = GF5_S2A_dataset(mat_save_path=dataset_dirs[args.dataset], kind='validate')
    elif args.dataset == 'WDCM':
        data_train = WDCM_dataset(mat_save_path=dataset_dir, kind='train', scale_factor=args.ratio)
        data_validate = WDCM_dataset(mat_save_path=dataset_dir, kind='validate', scale_factor=args.ratio)

    else:
        raise SystemExit('Error: no such type of dataset')

    if args.parallel:  # parallel training
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_validate)
        train_loader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=train_sampler)
        validate_loader = DataLoader(data_validate, batch_size=1, num_workers=args.num_workers, sampler=val_sampler)
    else:  # single GPU
        train_loader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        validate_loader = DataLoader(data_validate, batch_size=1, num_workers=args.num_workers, shuffle=True)

    dataloader = {'train': train_loader, 'eval': validate_loader}

    print(f'total training patches: {len(data_train)}')
    print(f'total evaluation patches: {len(data_validate)}')
    print(f'total training interation per epoch: {len(dataloader["train"])}')
    print(f'total evaluation interation: {len(dataloader["eval"])}')

    return dataloader


# test
if __name__ == '__main__':
    args = argsParser()
    dataloaders = get_dataloader(args)
    train_loader = dataloaders['train']
    for i, batch in enumerate(tqdm(train_loader)):
        for j in range(batch['lrhsi'].shape[0]):
            if torch.max(batch['lrhsi'][j]) == 0:
                print('最大值为0')
            if torch.max(batch['hrhsi'][j]) == 0:
                print('最大值为0')
            if torch.max(batch['hrmsi'][j]) == 0:
                print('最大值为0')
    print('数值正确')

