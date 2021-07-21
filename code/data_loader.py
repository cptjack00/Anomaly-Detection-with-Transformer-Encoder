import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.load_dataset(self.config['dataset'])
    
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
        self.rolling_windows = np.lib.stride_tricks.sliding_window_view(data['training'], self.config['l_win'], axis=0, writeable=True)