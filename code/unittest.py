from torch.utils.data import dataloader
from data_loader import CustomDataset
import torch
import numpy as np
from torch.autograd import Variable
from train import create_dataloader, window_mask
from model import make_model
import matplotlib.pyplot as plt
from model import PositionalEncoding

config = {'dataset': 'scada1_1', 'l_win': 500, 'pre_mask': 200, 'post_mask': 400,
          'batch_size': 1, 'shuffle': True, 'dataloader_num_workers': 4}


def test_dataloader():
    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)
    return dataloader


def test_dataset():
    dataset = CustomDataset(config)
    print(dataset[0])
    return dataset


def test_model():
    model = make_model(
        N=6, d_model=16, l_win=config['l_win'], d_ff=128, h=8, dropout=0.2)
    dataloader = test_dataloader()
    src_mask = window_mask(config['l_win'], config)
    for i, batch in enumerate(dataloader):
        out = model(batch['input'], src_mask)
        print(out)
        break


def test_pos_enc():
    dataloader = test_dataloader()
    for i, batch in enumerate(dataloader):
        plt.figure(figsize=(15, 5))
        pe = PositionalEncoding(16, 0)
        print("PE: ", pe.pe.size())
        y = pe.forward(batch['input'])
        plt.plot(np.arange(500), y[0, 2:5, :].transpose(0, 1).data.numpy())
        plt.legend(["dim %d" % p for p in [2, 3, 4]])
        plt.savefig('test.png')
        break


if __name__ == '__main__':
    test_model()
    # test_pos_enc()
