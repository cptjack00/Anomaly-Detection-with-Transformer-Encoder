import time

import torch
from torch.utils.data.dataloader import DataLoader, T

from data_loader import CustomDataset
from models import make_autoencoder_model, make_transformer_model
from utils import create_dirs, get_args, process_config, save_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataloader(dataset, config):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=bool(config['shuffle']), num_workers=config['dataloader_num_workers'])


def loss_backprop(criterion, out, targets):
    assert out.size(1) == targets.size(1)
    loss = criterion(out, targets)
    loss.backward()
    return loss


def create_mask(config):
    mask = torch.ones(1, config['l_win'], config['l_win'])
    mask[:, config['pre_mask']:config['post_mask'], :] = 0
    mask[:, :, config['pre_mask']:config['post_mask']] = 0
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0))
    return mask


min_trans_loss = float('inf')
best_trans_model = None


def trans_train_epoch(train_iter, model, autoencoder, criterion, mask, opt, epoch, config):
    global min_trans_loss, best_trans_model
    model.train()
    model.to(device)
    autoencoder.eval()
    encoder = autoencoder.encoder
    encoder.to(device)
    for i, batch in enumerate(train_iter):
        src = batch['input'].float()
        src.to(device)
        src = encoder(src)
        trg = batch['target'].float()
        trg.to(device)
        trg = encoder(trg)
        out = model(src, src_mask=mask)

        opt.zero_grad()
        loss = loss_backprop(criterion, out, trg)
        opt.step()
        if i % 10 == 0:
            print(i, loss)
    if loss < min_trans_loss:
        torch.save(model.state_dict(),
                   config['checkpoint_dir'] + f"best_trans_{epoch}.pth")
        torch.save(opt.state_dict(),
                   config['checkpoint_dir'] + f"optimizer_trans_{epoch}.pth")
        min_trans_loss = loss
        best_trans_model = f"best_trans_{epoch}.pth"
    if best_trans_model != None:
        return best_trans_model


min_auto_loss = float('inf')
best_auto_model = None


def autoencoder_train_epoch(train_iter, model, criterion, opt, epoch, config):
    global min_auto_loss, best_auto_model
    model.train()
    model.to(device)
    for i, batch in enumerate(train_iter):
        src = batch['input'].float()
        src.to(device)
        trg = batch['target'].float()
        trg.to(device)
        out = model(src)
        opt.zero_grad()
        loss = loss_backprop(criterion, out, trg)
        opt.step()
        if i % 10 == 0:
            print(i, loss)
    if loss < min_auto_loss:
        torch.save(model.state_dict(),
                   config['checkpoint_dir'] + f"best_autoencoder_{epoch}.pth")
        torch.save(opt.state_dict(),
                   config['checkpoint_dir'] + f"optimizer_autoencoder_{epoch}.pth")
        min_auto_loss = loss
        best_auto_model = f"best_autoencoder_{epoch}.pth"
    if best_auto_model != None:
        return best_auto_model


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        print(Ex)
        print("Missing or invalid arguments")
        exit(0)

    create_dirs(config['result_dir'], config['checkpoint_dir'])

    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)

    start = time.time()
    # Training the autoencoder
    autoencoder_model = make_autoencoder_model(
        seq_len=config['autoencoder_dims'], d_model=config['d_model'])
    autoencoder_model.float()
    model_opt = torch.optim.Adam(autoencoder_model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(config['auto_num_epoch']):
        config['best_auto_model'] = autoencoder_train_epoch(
            dataloader, autoencoder_model, criterion, model_opt, epoch, config)
    print("COMPLETED TRAINING THE AUTOENCODER")
    config['auto_train_time'] = (time.time() - start) / 60
    start = time.time()
    # Training the transformer model
    mask = create_mask(config)
    trans_model = make_transformer_model(
        N=6, d_model=dataset.rolling_windows.shape[-1], l_win=config['l_win'], d_ff=128, h=1, dropout=0.1)
    trans_model.float()
    model_opt = torch.optim.Adam(trans_model.parameters())
    autoencoder_model.load_state_dict(
        torch.load(config['checkpoint_dir'] + config['best_auto_model']))
    for epoch in range(config['trans_num_epoch']):
        config['best_trans_model'] = trans_train_epoch(
            dataloader, trans_model, autoencoder_model, criterion, mask, model_opt, epoch, config)
    print("COMPLETED TRAINING THE TRANSFORMER")
    config['trans_train_time'] = (time.time() - start) / 60
    save_config(config)


if __name__ == '__main__':
    main()
