import torch
import time
from model.ESFS import ESFS
from torch.utils.data import DataLoader
# from dataset.datasets import CaveDataset, HarvardDataset, WDCMDataset, YREDataset
from dataset.Cave_dataset import CaveDataset
from dataset.Harvard_dataset import Val_test_dataset as Harvard_val
from dataset.GF5_S2A_dataset import GF5_S2A_dataset
from dataset.WDCM_dataset import WDCM_dataset
from thop import profile
import shutil
from utils import Logger
import argparse
import scipy.io as scio
from tqdm import tqdm
import numpy as np
import os
from metrics import calc_psnr, calc_ssim, calc_sam, calc_rmse, calc_ergas, RASE

def prepare(sample_batched, device):
    for key in sample_batched.keys():
        sample_batched[key] = sample_batched[key].to(device)
    return sample_batched

def main():
    parser = argparse.ArgumentParser()

    ### log setting

    parser.add_argument('--save_dir', type=str, default='./test/cave/4/final',
                        help='Directory to save log, arguments, models and images')
    parser.add_argument('--reset', type=bool, default=True,
                        help='Delete save_dir to create a new one')
    parser.add_argument('--log_file_name', type=str, default='test.log',
                        help='Log file name')
    parser.add_argument('--logger_name', type=str, default='test',
                        help='Logger name')
    parser.add_argument('--cpu', type=bool, default=False,
                        help='Use CPU to run code')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='cave', choices=['cave', 'harvard', 'WDCM', 'YRE'],
                        help='Which dataset to train and test')
    parser.add_argument('--dataset_dir', type=str, default='/home3/mzq/data/Cave_mat/4/',
                        help='Directory of dataset')
    parser.add_argument('--ratio', type=int, default=4)
    parser.add_argument('--arch', type=str, default='ESFS',
                        choices=[
                            'SSRNET'
                                 ])
    ### model setting
    parser.add_argument('--hsi_channel', type=int, default=31)
    parser.add_argument('--msi_channel', type=int, default=3)


    parser.add_argument('--model_path', type=str, default='./train/cave/4/final/model/best.pt',
                        help='The path of model to evaluation')

    args = parser.parse_args()


    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ## make save_dir
    if (os.path.exists(args.save_dir)):
        if (not args.reset):
            raise SystemExit(
                'Error: save_dir "' + args.save_dir + '" already exists! Please set --reset True to delete the folder.')
        else:
            shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir)

    #os.makedirs(os.path.join(args.save_dir, 'save_results'))

    args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30, ' ') + '\t' + str(v) + '\n')

    logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name),
                     logger_name=args.logger_name).get_log()


    if (args.dataset == 'cave'):
        data_test = CaveDataset(args.dataset_dir, kind='test')
        testLoader = DataLoader(data_test, batch_size=1)
    elif (args.dataset == 'harvard'):
        data_test = Harvard_val(args.dataset_dir, kind='test')
        testLoader = DataLoader(data_test, batch_size=1)
    elif (args.dataset == 'WDCM'):
        data_test = WDCM_dataset(args.dataset_dir, kind='validate', scale_factor=args.ratio)
        testLoader = DataLoader(data_test, batch_size=1)
    elif (args.dataset == 'YRE'):
        data_test = GF5_S2A_dataset(args.dataset_dir, kind='eval')
        testLoader = DataLoader(data_test, batch_size=1)

    _model = ESFS(hsi_channel=args.hsi_channel, msi_channel=args.msi_channel, upscale_factor=args.ratio).to(device)

    logger.info('load_model_path: ' + args.model_path)
    model_state_dict_save = {k: v for k, v in torch.load(args.model_path, map_location=device).items()}
    model_state_dict = _model.state_dict()
    model_state_dict.update(model_state_dict_save)
    _model.load_state_dict(model_state_dict)

    times = []
    toc = time.time()
    ### test
    logger.info('Test process...')
    psnr = []
    ssim = []
    sam = []
    ergas = []
    rmse = []
    rase = []
    _model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(tqdm(testLoader)):
            sample_batched = prepare(sample_batched, args.device)
            hrhsi = sample_batched['hrhsi']
            hrmsi = sample_batched['hrmsi']
            lrhsi = sample_batched['lrhsi']

            with torch.no_grad():
                # sr = _model(lrhsi, hrmsi)
                sr = _model(lrhsi, hrmsi)
                
            sr = torch.clamp(sr, 0.0, 1.0)
            psnr.append(calc_psnr(hrhsi.detach(), sr.detach()))
            ssim.append(calc_ssim(hrhsi.detach(), sr.detach()))
            sam.append(calc_sam(hrhsi.detach(), sr.detach()))
            ergas.append(calc_ergas(hrhsi.detach(), sr.detach()))
            rmse.append(calc_rmse(hrhsi.detach(), sr.detach()))
            rase.append(RASE(hrhsi.detach().cpu().numpy(), sr.detach().cpu().numpy(), bands=hrhsi.shape[1]))


            sr = (np.transpose(sr.cpu().numpy(), (0, 2, 3, 1)))


            save_path = os.path.join(args.save_dir, 'img{}.mat'.format(i_batch))
            scio.savemat(save_path, {'fus': sr[0]})

        psnr_ave = np.mean(psnr)
        ssim_ave = np.mean(ssim)
        sam_ave = np.mean(sam)
        ergas_ave = np.mean(ergas)
        rmse_ave = np.mean(rmse)
        rase_avg = np.mean(rase)

        flops, params = profile(_model, (lrhsi, hrmsi))

        logger.info(
            'Test PSNR (now): %.6f \t SSIM (now): %.6f \t SAM (now): %.6f \t ERGAS (now): %.6f \t RMSE (now): %.6f \t RASE (now): %.6f \t flops (now): %.6f \t params (now): %.6f' % (
            psnr_ave, ssim_ave, sam_ave, ergas_ave, rmse_ave, rase_avg, flops/1e12, params/1000000.0))
        logger.info('Test over.')

    tic = time.time()
    times.append(tic - toc)
    print("Running Time: {:.6f}".format(np.mean(times)))


if __name__ == '__main__':
    main()
