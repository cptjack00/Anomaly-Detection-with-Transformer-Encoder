import time

import torch
from torch.utils.data.dataloader import DataLoader

from data_loader import CustomDataset
from models import make_autoencoder_model, make_trans_model
from utils import create_dirs, get_args, process_config, save_config

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and not torch.cuda.is_initialized():
    torch.cuda.init()

def create_dataloader(dataset, config):
    return DataLoader(dataset,
                      batch_size=config["batch_size"],
                      shuffle=bool(config["shuffle"]),
                      num_workers=config["dataloader_num_workers"])


def loss_backprop(criterion, out, targets):
    assert out.size(1) == targets.size(1)
    loss = criterion(out, targets)
    loss.backward()
    return loss


def create_mask(config):
    mask = torch.ones(1, config["l_win"], config["l_win"])
    mask[:, config["pre_mask"]:config["post_mask"], :] = 0
    mask[:, :, config["pre_mask"]:config["post_mask"]] = 0
    mask = mask.float().masked_fill(mask == 0, float(
        "-inf")).masked_fill(mask == 1, float(0))
    return mask


min_vanilla_loss = float("inf")
best_vanilla_model = None
epoch_vanilla_loss = list()

def vanilla_train_epoch(train_iter, model, autoencoder, criterion, mask, opt, epoch, config):
    global min_vanilla_loss, best_vanilla_model, epoch_vanilla_loss
    model.train()
    model.to(device)
    autoencoder.eval()
    encoder = autoencoder.encoder
    encoder.to(device)
    batch_loss = list()
    for i, batch in enumerate(train_iter):
        src = batch["input"].float()
        src = src.to(device)
        src = encoder(src)
        trg = batch["target"].float()
        trg = trg.to(device)
        trg = encoder(trg)
        out = model(src, src_mask=mask)

        opt.zero_grad()
        loss = loss_backprop(criterion, out, trg)
        opt.step()
        batch_loss.append(loss.item())

    if len(batch_loss) > 0:
        epoch_vanilla_loss.append(sum(batch_loss)/len(batch_loss))
        print('VANILLA TRANSFORMER. Epoch: {} \tTotal Loss: {:.6f}'.format(epoch,
                                                                          epoch_vanilla_loss[-1]))

    if epoch_vanilla_loss[-1] < min_vanilla_loss:
        torch.save(model.state_dict(),
                   config["checkpoint_dir"] + f"best_vanilla_{epoch}.pth")
        torch.save(opt.state_dict(),
                   config["checkpoint_dir"] + f"optimizer_vanilla_{epoch}.pth")
        min_vanilla_loss = epoch_vanilla_loss[-1]
        best_vanilla_model = f"best_vanilla_{epoch}.pth"
    if best_vanilla_model != None:
        return best_vanilla_model


min_auto_loss = float("inf")
best_auto_model = None
epoch_auto_loss = list()

def autoencoder_train_epoch(train_iter, model, criterion, opt, epoch, config):
    global min_auto_loss, best_auto_model, epoch_auto_loss
    model.train()
    model.to(device)
    batch_loss = list()
    for i, batch in enumerate(train_iter):
        src = batch["input"].float()
        src = src.to(device)
        trg = batch["target"].float()
        trg = trg.to(device)
        out = model(src)
        opt.zero_grad()
        loss = loss_backprop(criterion, out, trg)
        opt.step()
        batch_loss.append(loss.item())

    if len(batch_loss) > 0:
        epoch_auto_loss.append(sum(batch_loss)/len(batch_loss))
        print('AUTOENCODER. Epoch: {} \tTotal Loss: {:.6f}'.format(epoch,
                                                                          epoch_auto_loss[-1]))

    if epoch_auto_loss[-1] < min_auto_loss:
        torch.save(model.state_dict(),
                   config["checkpoint_dir"] + f"best_autoencoder_{epoch}.pth")
        torch.save(opt.state_dict(),
                   config["checkpoint_dir"] + f"optimizer_autoencoder_{epoch}.pth")
        min_auto_loss = epoch_auto_loss[-1]
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

    create_dirs(config["result_dir"], config["checkpoint_dir"])

    dataset = CustomDataset(config)
    dataloader = create_dataloader(dataset, config)

    # Training the autoencoder
    start = time.time()
    autoencoder_model = make_autoencoder_model(in_seq_len=config["autoencoder_dims"],
                                               out_seq_len=config["l_win"],
                                               d_model=config["d_model"])
    autoencoder_model.float()
    model_opt = torch.optim.Adam(autoencoder_model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(config["auto_num_epoch"]):
        config["best_auto_model"] = autoencoder_train_epoch(dataloader,
                                                            autoencoder_model,
                                                            criterion,
                                                            model_opt,
                                                            epoch,
                                                            config)
    print("COMPLETED TRAINING THE AUTOENCODER")
    config["auto_train_time"] = (time.time() - start) / 60

    # Training the model
    start = time.time()
    mask = create_mask(config)
    vanilla_model = make_trans_model(N=config["num_stacks"],
                                         d_model=config["d_model"],
                                         l_win=config["l_win"],
                                         d_ff=config["d_ff"],
                                         h=config["num_heads"],
                                         dropout=config["dropout"])
    vanilla_model.float()
    model_opt = torch.optim.Adam(vanilla_model.parameters())
    autoencoder_model.load_state_dict(
        torch.load(config["checkpoint_dir"] + config["best_auto_model"]))
    for epoch in range(config["trans_num_epoch"]):
        config["best_vanilla_model"] = vanilla_train_epoch(dataloader,
                                                       vanilla_model,
                                                       autoencoder_model,
                                                       criterion,
                                                       mask,
                                                       model_opt,
                                                       epoch,
                                                       config)
    print("COMPLETED TRAINING THE VANILLA TRANSFORMER")
    config["vanilla_train_time"] = (time.time() - start) / 60
    save_config(config)

if __name__ == "__main__":
    main()
