import torch
from torch.utils.data.dataloader import DataLoader
from data_loader import CustomDataset
from model import make_model

config = {'dataset': 'scada1_1', 'l_win': 200, 'pre_mask': 80, 'post_mask': 120,
          'batch_size': 32, 'shuffle': True, 'dataloader_num_workers': 4, 'num_epoch': 50, 'model_path': '../models/'}


def create_dataloader(dataset, config):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['dataloader_num_workers'])


def loss_backprop(criterion, out, targets):
    assert out.size(1) == targets.size(1)
    loss = criterion(out, targets)
    loss.backward()
    return loss

def create_mask(config):
    mask = torch.ones(1, config['l_win'], config['l_win'])
    mask[:, config['pre_mask']:config['post_mask'], :] = 0
    mask[:, :, config['pre_mask']:config['post_mask']] = 0
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
    return mask


def train_epoch(train_iter, model, criterion, mask, opt, min_train_loss):
    model.train()
    for i, batch in enumerate(train_iter):
        src = batch['input'].float()
        trg = batch['target'].float()
        # the words we are trying to predict
        out = model(src, src_mask=mask)

        opt.zero_grad()
        loss = loss_backprop(criterion, out, trg)
        opt.step()
        if i % 10 == 0:
            print(i, loss)
    if loss < min_train_loss:
        torch.save(model.state_dict(), config['model_path'] + f"best_train_{epoch}.pth")
        torch.save(opt.state_dict(), config['model_path'] + f"optimizer_{epoch}.pth")
        min_train_loss = loss
        best_model = f"best_train_{epoch}.pth"
    if best_model != None:
        return best_model

if __name__ == '__main__':
    min_train_loss = float('inf')
    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)
    mask = create_mask(config)
    model = make_model(
        N=6, d_model=dataset.rolling_windows.shape[-1], l_win=config['l_win'], d_ff=128, h=1, dropout=0.1)
    model.float()
    model_opt = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(config['num_epoch']):
        train_epoch(dataloader, model, criterion, mask, model_opt, min_train_loss)
