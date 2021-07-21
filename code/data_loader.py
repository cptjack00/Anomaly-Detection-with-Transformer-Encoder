import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.load_dataset(self.config['dataset'])
        self.input_mask = self._create_window_mask()

    def __len__(self):
        return self.rolling_windows.shape[0]

    def __getitem__(self, index):
        input = self.rolling_windows[index, :, :]
        sample = {"input": input, "target": input}
        return sample

    def load_dataset(self, dataset):
        data_dir = '../data/scada/'
        data = np.load(data_dir + dataset + '.npz')

        # normalise the dataset by training set mean and std
        # train_m = data['train_m']
        # train_std = data['train_std']
        # readings_normalised = (data['readings'] - train_m) / train_std

        # slice training set into rolling windows
        self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
            data['training'], self.config['l_win'], axis=0, writeable=True).transpose(0, 2, 1)

    def _create_window_mask(self):
        """Mask out the defined positions."""
        config = self.config
        attn_shape = (1, config['l_win'], self.rolling_windows.shape[-1])
        window_mask = np.ones(attn_shape).astype('float')
        window_mask[:, config['pre_mask']:config['post_mask'], :] = 0.0
        return torch.from_numpy(window_mask)

    def mask_input(self):
        self.input_mask = self.input_mask.masked_fill(self.input_mask == 0, float(-1e9)).masked_fill(self.input_mask == 1, 0)
        self.rolling_windows = np.add(self.rolling_windows, self.input_mask)
