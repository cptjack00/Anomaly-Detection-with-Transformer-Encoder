import torch
import torch
from torch.utils.data.dataloader import DataLoader
from data_loader import CustomDataset
from model import make_model

config = {'dataset': 'scada1_1', 'l_win': 500, 'pre_mask': 200, 'post_mask': 400,
          'batch_size': 32, 'shuffle': True, 'dataloader_num_workers': 4, 'num_epoch': 20, 'model_path': '../models/'}


def create_dataloader(dataset, config):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['dataloader_num_workers'])


def loss_backprop(criterion, out, targets):
    assert out.size(1) == targets.size(1)
    total = 0.0
    loss = criterion(out, targets)
    total += loss
    loss.backward()
    return total


def train_epoch(train_iter, model, criterion, opt, min_train_loss):
    model.train()
    for i, batch in enumerate(train_iter):
        src = batch['input'].float()
        trg = batch['target'].float()
        # the words we are trying to predict
        out = model(src, src_mask=None)

        opt.zero_grad()
        loss = loss_backprop(criterion, out, trg)
        opt.step()
        if i % 10 == 0:
            print(i, loss)
        break
    if loss < min_train_loss:
        torch.save(model.state_dict(), config['model_path'] + f"best_train_{epoch}.pth")
        torch.save(opt.state_dict(), config['model_path'] + f"optimizer_{epoch}.pth")
        min_train_loss = loss
        best_model = f"best_train_{epoch}.pth"
    if best_model != None:
        return best_model

min_train_loss = float('inf')
dataset = CustomDataset(config)
dataloader = create_dataloader(dataset, config)
model = make_model(
    N=6, d_model=dataset.rolling_windows.shape[-1], l_win=config['l_win'], d_ff=128, h=1, dropout=0.1)
model.float()
model_opt = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()
for epoch in range(config['num_epoch']):
    train_epoch(dataloader, model, criterion, model_opt, min_train_loss)
