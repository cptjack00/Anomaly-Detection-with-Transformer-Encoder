import torch
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from data_loader import CustomDataset
from model import make_model
from utils import NoamOpt, loss_backprop

config = {'dataset': 'scada1_1', 'l_win': 500, 'pre_mask': 200, 'post_mask': 400,
          'batch_size': 1, 'shuffle': True, 'dataloader_num_workers': 4}

def create_dataloader(dataset, config):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['dataloader_num_workers'])

if __name__ == "__main__":
    model = make_model(
        N=6, d_model=16, l_win=config['l_win'], d_ff=128, h=8, dropout=0.2)
    model.float()
    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)
    for i, batch in enumerate(dataloader):
        out = model(batch['input'].float(), src_mask=None)
        print(out)
        break

