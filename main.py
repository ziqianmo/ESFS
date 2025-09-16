import torch
import argsParser
from utils import mkExpDir
from model.ESFS import ESFS
from trainer import Trainer
import torch.optim as optim
from torch import nn
from dataset import dataloader
from utils import *
from torch.optim.lr_scheduler import MultiStepLR

args = argsParser.argsParser()
print(args)

def load(model, model_path=None, device=None):
        if (model_path):
            model_state_dict_save = {k: v for k, v in torch.load(model_path, map_location=device).items()}
            model_state_dict = model.state_dict()
            model_state_dict.update(model_state_dict_save)
            model.load_state_dict(model_state_dict)

def main():
    SEED = 110401
    torch.manual_seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ### make save_dir
    _logger = mkExpDir(args)

    _dataloader = dataloader.get_dataloader(args)

    _model = ESFS(hsi_channel=args.hsi_channel, msi_channel=args.msi_channel, upscale_factor=args.ratio).to(device)


    optimizer = optim.Adam(_model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma)
    

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, device, optimizer)

    ###  train

    for epoch in range(1, args.num_epochs + 1):
        t.train(current_epoch=epoch)
        print(epoch)
        if (epoch % args.val_every == 0):
            t.evaluate(current_epoch=epoch)
        scheduler.step()

if __name__ == '__main__':
    main()
