from torch.utils.data import dataloader
from data_loader import CustomDataset
import torch
import numpy as np
from torch.autograd import Variable
from train import create_dataloader
from utils import process_config, create_dirs, get_args, save_config
from models import make_transformer_model, PositionalEncoding, make_autoencoder_model
import matplotlib.pyplot as plt

config = {'dataset': 'scada1_1', 'l_win': 200, 'pre_mask': 80, 'post_mask': 120,
          'batch_size': 1, 'shuffle': True, 'dataloader_num_workers': 4, "data_dir": "scada"}


def test_dataloader():
    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)
    return dataloader


def test_dataset():
    dataset = CustomDataset(config)
    dataset.mask_input()
    print(dataset[0]['input'].shape)
    print(dataset[0]['input'][199: 205, :])
    return dataset


def test_model():
    model = make_transformer_model(
        N=6, d_model=16, l_win=config['l_win'], d_ff=128, h=8, dropout=0.2)
    model.float()
    dataloader = test_dataloader()
    for i, batch in enumerate(dataloader):
        out = model(batch['input'].float(), src_mask=None)
        print(out)
        break


def test_pos_enc():
    dataloader = test_dataloader()
    for i, batch in enumerate(dataloader):
        plt.figure(figsize=(16, 5))
        pe = PositionalEncoding(16, 0)
        print("PE: ", pe.pe.size())
        y = pe.forward(torch.zeros_like(batch['input']))
        plt.plot(np.arange(config['l_win']), y[0, :, 2:5].data.numpy())
        plt.legend(["dim %d" % p for p in [2, 3, 4]])
        plt.savefig('test.png')
        break

def test_autoencoder():
    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)
    model = make_autoencoder_model(config['l_win'] * 16)
    for i, batch in enumerate(dataloader):
        print(batch['input'].shape)
        out = model(batch['input'].float())
        print(out.shape)
        break
    # for i, batch in enumerate(dataloader):
    #     plt.figure(figsize=(16, 5))
    #     pe = PositionalEncoding(16, 0)
    #     print("PE: ", pe.pe.size())
    #     y = pe.forward(torch.zeros_like(batch['input']))
    #     plt.plot(np.arange(config['l_win']), y[0, :, 2:5].data.numpy())
    #     plt.legend(["dim %d" % p for p in [2, 3, 4]])
    #     plt.savefig('test.png')
    #     break


if __name__ == '__main__':
    # test_model()
    # test_dataset()
    # test_pos_enc()
    test_autoencoder()

