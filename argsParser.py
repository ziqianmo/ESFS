import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argsParser():
    parser = argparse.ArgumentParser()


    ### log setting
    parser.add_argument('--save_dir', type=str, default='./train/cave/4/final',
                        help='Directory to save log, arguments, models and images')
    parser.add_argument('--reset', type=str2bool, default=True,
                        help='Delete save_dir to create a new one')
    parser.add_argument('--log_file_name', type=str, default='train.log',
                        help='Log file name')
    parser.add_argument('--logger_name', type=str, default='train',
                        help='Logger name')
    parser.add_argument('--arch', type=str, default='ESFS',
                        choices=[
                            'SSRNET', 'MSSJFL','DHIFNET'
                                 ])

    ### hsi msi device setting
    parser.add_argument('--cpu', type=str2bool, default=False,
                        help='Use CPU to run code')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='The number of GPU used in training')

    ### dataset setting
    parser.add_argument('--dataset', type=str, default='cave', choices=['cave', 'harvard', 'WDCM', 'YRE'],
                        help='Which dataset to train and test')
    parser.add_argument('--ratio', type=int, default=4)

    ### dataloader setting
    parser.add_argument('--num_workers', type=int, default=1,
                        help='The number of workers when loading data')

    ### optimizer setting
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--step', type=list, default=[100, 150, 175, 190, 195],
                        help='step size for step decay')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Learning rate decay factor for step decay')

    ### model setting
    parser.add_argument('--hsi_channel', type=int, default=31)
    parser.add_argument('--msi_channel', type=int, default=3)


    ### training setting
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='The number of training epochs')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save period')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Validation period')
    parser.add_argument('--test', type=str2bool, default=False,
                        help='Test mode')
    parser.add_argument('--parallel',type=str2bool, default=False,
                        help='parallel')


    args = parser.parse_args()

    return args
