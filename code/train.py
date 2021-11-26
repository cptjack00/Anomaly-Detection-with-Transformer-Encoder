import time
import os
import torch
from torch.utils.data.dataloader import DataLoader

from data_loader import CustomDataset
from models import make_autoencoder_model, make_fnet_hybrid_model
from utils import create_dirs, get_args, process_config, save_config
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trans_var = "FNET-HYBRID"
if device.type == "cuda" and not torch.cuda.is_initialized():
    torch.cuda.init()


def plot_loss(loss_train, loss_val, epochs, model, config):
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('{}: Training and Validation Loss'.format(model))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config["result_dir"], model))
    plt.close()


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



def hybrid_train(train_iter, val_iter, model, autoencoder, criterion, mask, opt, num_epochs, config):
    min_hybrid_loss = float("inf")
    best_hybrid_model = None
    best_hybrid_optimizer = None
    model.train()
    model.to(device)
    autoencoder.eval()
    encoder = autoencoder.encoder
    encoder.to(device)
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, 1 + num_epochs):
        train_loss = 0.0
        model.train()
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
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        for i, batch in enumerate(val_iter):
            src = batch["input"].float()
            src = src.to(device)
            src = encoder(src)
            trg = batch["target"].float()
            trg = trg.to(device)
            trg = encoder(trg)
            out = model(src, src_mask=mask)
            loss = criterion(out, trg)
            val_loss += loss.item()
            
        train_loss = train_loss / len(train_iter)
        val_loss = val_loss / len(val_iter)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print('{}. Epoch: {} | Train Loss: {:.6f}'.format(trans_var, epoch, train_loss))
        print('{}. Epoch: {} | Validation Loss: {:.6f}'.format(trans_var, epoch, val_loss))

        if val_loss < min_hybrid_loss:
            best_hybrid_model = model.state_dict()
            best_hybrid_optimizer = opt.state_dict()
            min_hybrid_loss = val_loss
    plot_loss(train_loss_list, val_loss_list, range(1, num_epochs + 1), trans_var, config)
    torch.save(best_hybrid_model,
               config["checkpoint_dir"] + f"best_hybrid.pth")
    torch.save(best_hybrid_optimizer,
               config["checkpoint_dir"] + f"optimizer_hybrid.pth")
    return "best_hybrid.pth"


def autoencoder_train(train_iter, val_iter, model, criterion, opt, num_epochs, config):
    min_auto_loss = float("inf")
    best_auto_model = None
    best_auto_optimizer = None
    model.train()
    model.to(device)
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, 1 + num_epochs):
        train_loss = 0.0
        model.train()
        for i, batch in enumerate(train_iter):
            src = batch["input"].float()
            src = src.to(device)
            trg = batch["target"].float()
            trg = trg.to(device)
            out = model(src)
            opt.zero_grad()
            loss = loss_backprop(criterion, out, trg)
            opt.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        for i, batch in enumerate(val_iter):
            src = batch["input"].float()
            src = src.to(device)
            trg = batch["target"].float()
            trg = trg.to(device)
            out = model(src)
            loss = criterion(out, trg)
            val_loss += loss.item()
        train_loss = train_loss / len(train_iter)
        val_loss = val_loss / len(val_iter)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print('AUTOENCODER. Epoch: {} | Train Loss: {:.6f}'.format(epoch, train_loss))
        print('AUTOENCODER. Epoch: {} | Validation Loss: {:.6f}'.format(epoch, val_loss))

        if val_loss < min_auto_loss:
            min_auto_loss = val_loss
            best_auto_model = model.state_dict()
            best_auto_optimizer = opt.state_dict()
    plot_loss(train_loss_list, val_loss_list, range(1, num_epochs + 1), "AUTOENCODER", config)
    torch.save(best_auto_model,
               config["checkpoint_dir"] + f"best_auto.pth")
    torch.save(best_auto_optimizer,
               config["checkpoint_dir"] + f"optimizer_auto.pth")
    return "best_auto.pth"


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        print(Ex)
        print("Missing or invalid arguments")
        exit(0)

    create_dirs(config["result_dir"], config["checkpoint_dir"])

    train_set = CustomDataset(config, mode='train')
    train_dataloader = create_dataloader(train_set, config)
    val_set = CustomDataset(config, mode='valid')
    val_dataloader = create_dataloader(val_set, config) 

    # Training the autoencoder
    start = time.time()
    autoencoder_model = make_autoencoder_model(in_seq_len=config["autoencoder_dims"],
                                               out_seq_len=config["l_win"],
                                               d_model=config["d_model"])
    autoencoder_model.float()
    model_opt = torch.optim.Adam(autoencoder_model.parameters(), lr=config["lr"])
    criterion = torch.nn.MSELoss()
    config["best_auto_model"] = autoencoder_train(train_dataloader,
                                                        val_dataloader,
                                                        autoencoder_model,
                                                        criterion,
                                                        model_opt,
                                                        config["auto_num_epoch"], 
                                                        config)
    print("COMPLETED TRAINING THE AUTOENCODER")
    config["auto_train_time"] = (time.time() - start) / 60

    # Training the FNet-Hybrid model
    start = time.time()
    mask = create_mask(config)
    fnet_hybrid_model = make_fnet_hybrid_model(N=config["num_stacks"],
                                         d_model=config["d_model"],
                                         l_win=config["l_win"],
                                         d_ff=config["d_ff"],
                                         h=config["num_heads"],
                                         dropout=config["dropout"])
    fnet_hybrid_model.float()
    model_opt = torch.optim.Adam(fnet_hybrid_model.parameters(), lr=config["lr"])
    autoencoder_model.load_state_dict(torch.load(config["checkpoint_dir"] + config["best_auto_model"]))
    config["best_hybrid_model"] = hybrid_train(train_dataloader,
                                                     val_dataloader,
                                                     fnet_hybrid_model,
                                                     autoencoder_model,
                                                     criterion,
                                                     mask,
                                                     model_opt,
                                                     config["trans_num_epoch"],
                                                     config)
    print("COMPLETED TRAINING THE FNET-HYBRID")
    config["hybrid_train_time"] = (time.time() - start) / 60
    print(config)
    save_config(config)

if __name__ == "__main__":
    main()
