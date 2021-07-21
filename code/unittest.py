from data_loader import CustomDataset
from train import create_dataloader, window_mask

config = {'dataset': 'scada1_1', 'l_win': 500, 'pre_mask': 200, 'post_mask': 400,
          'batch_size': 32, 'shuffle': True, 'dataloader_num_workers': 4}


def test_dataloader():
    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)
    for i, sample in enumerate(dataloader):
        print(sample['target'].size())
        print(sample['input'].size())
        break
    return dataloader

def test_dataset():
    dataset = CustomDataset(config)
    print(dataset[0])
