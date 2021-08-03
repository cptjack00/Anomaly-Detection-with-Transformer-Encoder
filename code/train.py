import torch
from utils import process_config, create_dirs, get_args, save_config
from torch.utils.data.dataloader import DataLoader, T
from data_loader import CustomDataset
from models import make_transformer_model, make_autoencoder_model


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
    for i, batch in enumerate(train_iter):
        src = batch['input'].float()
        src = autoencoder(src)
        trg = batch['target'].float()
        trg = autoencoder(src)
        # the words we are trying to predict
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
    for i, batch in enumerate(train_iter):
        src = batch['input'].float()
        trg = batch['target'].float()
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
    except:
        print("Missing or invalid arguments")
        exit(0)

    create_dirs(config['result_dir'], config['checkpoint_dir'])

    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)

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
    # Training the transformer model
    mask = create_mask(config)
    trans_model = make_transformer_model(
        N=6, d_model=dataset.rolling_windows.shape[-1], l_win=config['l_win'], d_ff=128, h=1, dropout=0.1)
    trans_model.float()
    best_auto_model = autoencoder_model.load_state_dict(torch.load(config['model_path'] + config['best_auto_model']))
    for epoch in range(config['trans_num_epoch']):
        config['best_trans_model'] = trans_train_epoch(
            dataloader, trans_model, best_auto_model, criterion, mask, model_opt, epoch, config)
    print("COMPLETED TRAINING THE TRANSFORMER")

    save_config(config)


if __name__ == '__main__':
    main()
