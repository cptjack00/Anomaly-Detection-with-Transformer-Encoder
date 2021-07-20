import numpy as np


class DataGenerator():
    def __init__(self, config):
        self.config = config
        # load data here: generate 3 state variables: train_set, val_set and test_set
        self.load_NAB_dataset(self.config['dataset'])

    def load_dataset(self, dataset):
        data_dir = '../datasets/scada1/'
        data = np.load(data_dir + dataset + '.npz')

        # normalise the dataset by training set mean and std
        # train_m = data['train_m']
        # train_std = data['train_std']
        # readings_normalised = (data['readings'] - train_m) / train_std

        # slice training set into rolling windows
        rolling_windows = np.lib.stride_tricks.sliding_window_view(data['training'], self.config['l_win'], axis=0, writable=False)
        return rolling_windows


