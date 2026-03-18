import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import Subset
import importlib.util
import argparse

from local_datasets.datasets import get_dataloaders
from continuous_dropouts import *
from estimators.doe_util import DoE
from estimators.club_util import CLUB
from estimators.mine import MINE

# EXP_PATH = Path('checkpoints/densenet_cifar100/optuna_lambda_lmbd0.001')
DATA_DIM = 3*32*32 # experiment specific! 3*32*32 for CIFAR and SVHN, 28*28 for FMNIST, 256 for mini-Bert embeddings
REPR_DIM = 128 # experiment specific!

#BINS = 5
critic_params = {
    'layers': 2,
    'hidden': 128 # align with REPR_DIM
}
SAMPLES = 4
LR = 0.0001
GRAD_CLIP = 1
BS = 256
EPOCHS = 3

def extract_representations(model, dataloader, device='cuda', llm_mode=False):
    reps = []
    labels = []
    with torch.no_grad():
        for B in dataloader:
            if llm_mode:
                input_ids = B["input_ids"].to(device)
                attention_mask = B["attention_mask"].to(device)
                y = B["label"].to(device)
                r = model.representation(input_ids=input_ids, attention_mask=attention_mask)
            else:
                X, y = B
                X, y = X.to(device), y.to(device)
                r = model.representation(X)        # (B, REPR_DIM), on device
            reps.append(r)
            labels.append(y)
    reps = torch.cat(reps, dim=0)   # (N, D)
    labels = torch.cat(labels, dim=0)
    return reps, labels

# single batch training for DoE with SAMPLES stochastic reps
def train_doe(doe, optimizer, model, loader, device, llm_mode=False):
    doe.train()
    mi_loss_log = []
    for step, B in enumerate(loader):
        if llm_mode:
            input_ids = B["input_ids"].to(device)
            attention_mask = B["attention_mask"].to(device)
            curX = model.bert.embeddings(input_ids)
        else:
            batch_X, _ = B
            batch_X = batch_X.to(device)  # (B, C, H, W)
            # flatten samples once on GPU
            curX = batch_X #.flatten(start_dim=1)  # (B, DATA_DIM)
        # collect SAMPLES representations on GPU
        losses = []
        # we have to avoid passing the same example with multiple noisy representations,
        # because then DoE will match negative pairs wrongly (same X with another representation is not negative)
        for _ in range(SAMPLES):
            # enable train mode for model if you want stochasticity (dropout)
            if llm_mode:
                r = model.representation(input_ids=input_ids, attention_mask=attention_mask).detach()
            else:
                r = model.representation(batch_X).detach()    # (B, REPR_DIM) on device
            loss = doe(curX, r)
            losses.append(loss)
        # Average the losses over SAMPLES
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(doe.parameters(), GRAD_CLIP)
        optimizer.step()

        mi_loss_log.append(total_loss.item())

        if (step+1) % 10 == 0:
            print('step {:4d} | '.format(step+1), end='')
            print('doe_l: {:6.2f} | '.format(-loss.item()))
    return loss, mi_loss_log

def train_club(club, optimizer, model, loader, device):
    club.train()
    mi_est = []
    for step, B in enumerate(loader):
        batch_X, _ = B
        batch_X = batch_X.to(device)  # (B, C, H, W)
        # flatten samples once on GPU
        curX = batch_X.flatten(start_dim=1)  # (B, DATA_DIM)
        losses = []
        for _ in range(SAMPLES):
            # enable train mode for model if you want stochasticity (dropout)
            r = model.representation(batch_X).detach()    # (B, REPR_DIM) on device
            loss = club.learning_loss(curX, r)
            losses.append(loss)
        # Average the losses over SAMPLES
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(club.parameters(), GRAD_CLIP)
        optimizer.step()

        with torch.no_grad():
            mi_est.append(club(curX, r).item())

        if (step+1) % 10 == 0:
            print('step {:4d} | '.format(step+1), end='')
            print('est: {:6.2f} | '.format(mi_est[-1]))
    return mi_est

def train_mine(mine, optimizer, model, loader, device):
    mine.train()
    mi_est = []
    for step, B in enumerate(loader):
        batch_X, _ = B
        batch_X = batch_X.to(device)  # (B, C, H, W)
        # flatten samples once on GPU
        curX = batch_X.flatten(start_dim=1)  # (B, DATA_DIM)
        losses = []
        for _ in range(SAMPLES):
            # enable train mode for model if you want stochasticity (dropout)
            r = model.representation(batch_X).detach()    # (B, REPR_DIM) on device
            mi, joint_mean, marg_exp_mean = mine(curX, r)
            loss = -mi
            losses.append(loss)
        # Average the losses over SAMPLES
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mine.parameters(), 5.0)
        optimizer.step()
        mi_est.append(mi.item())

        if (step+1) % 10 == 0:
            print('step {:4d} | '.format(step+1), end='')
            print('est: {:6.2f} | '.format(mi_est[-1]))
    return mi_est

def plot_mi_loss_log(loss_log, plot_name, smoothing=50):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_log, alpha=0.7, label="DoE loss (raw)")

    if len(loss_log) > smoothing:
        # simple moving average for smoothing
        smoothed = [
            sum(loss_log[i - smoothing:i]) / smoothing
            for i in range(smoothing, len(loss_log))
        ]
        plt.plot(range(smoothing, len(loss_log)), smoothed, c="red", label=f"smoothed ({smoothing})")

    plt.xlabel("Training step")
    plt.ylabel("DoE loss")
    plt.title("DoE critic training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP_PATH / (plot_name+".jpg"))
    plt.close()

def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, GaussianDropout)):
            m.train()

def report_MI(doe_loss, eps=1e-6):
    """
    Convert DoE loss to MI for reporting.
    - If loss is slightly positive (numerical noise), report 0.
    - If loss is significantly positive, report NaN (numerical instability).
    - Otherwise, return -loss.
    """
    value = doe_loss.cpu().item()  # scalar
    if value > eps:
        return float('nan')
    elif value > 0:  # tiny positive values
        return 0.0
    else:
        return -value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='exp_dir', type=str, help='experiment directory')
    parser.add_argument('--llm_mode', dest='llm_mode', type=str, help='is it for a text task transformer?', default="False")
    args = parser.parse_args()
    EXP_PATH = Path(args.exp_dir)
    args.llm_mode = args.llm_mode.lower() == "true"

    spec = importlib.util.spec_from_file_location("param_setup", "params_resenet18_cifar10.py")
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loaders = get_dataloaders(params.dset_dir, BS, drop_last=False)
    train_data = data_loaders['train'].dataset
    test_data = data_loaders['test'].dataset
    train_hist = pickle.load(open((EXP_PATH/"train_history.pkl"), "rb"))
    test_hist = pickle.load(open((EXP_PATH/"test_history.pkl"), "rb"))
    last_epoch = train_hist['epoch'][-1]
    params.model.load_state_dict(torch.load(EXP_PATH/("chkp_"+str(last_epoch)), weights_only=True))
    params.model.to(device)
    enable_dropout(params.model)

    print("MI for train data")
    ## MINE
    mine = MINE(DATA_DIM, REPR_DIM, hidden=min(512, 2 * REPR_DIM)).to(device)
    optimizer = optim.AdamW(mine.parameters(), lr=LR)
    mi_critic_logs = []
    for ep in range(EPOCHS):
        logs = train_mine(mine, optimizer, params.model, data_loaders['train'], device)
        mi_critic_logs += logs
    train_mi_xz = logs[-1]
    plot_mi_loss_log(mi_critic_logs, "MINE_train_data")
    ## CLUB
    #club = CLUB(DATA_DIM, REPR_DIM, min(512, 2 * REPR_DIM)).to(device)
    #optimizer = optim.AdamW(club.parameters(), lr=LR)
    #mi_critic_logs = []
    #for ep in range(EPOCHS):
    #    logs = train_club(club, optimizer, params.model, data_loaders['train'], device)
    #    mi_critic_logs += logs
    #train_mi_xz = logs[-1]
    #plot_mi_loss_log(mi_critic_logs, "CLUB_train_data")
    ## DoE
    #doe_l = DoE(DATA_DIM, REPR_DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)
    #optimizer = optim.AdamW(doe_l.parameters(), lr=LR)
    #mi_critic_logs = []
    #for ep in range(EPOCHS):
    #    L_doe_l, logs = train_doe(doe_l, optimizer, params.model, data_loaders['train'], device, args.llm_mode)
    #    mi_critic_logs += logs
    #train_mi_xz = report_MI(L_doe_l)
    #plot_mi_loss_log(mi_critic_logs, "DoE_train_data")
    print("Full MI train ", train_mi_xz)

    print("MI for test data")
    ## MINE
    mine_test = MINE(DATA_DIM, REPR_DIM, hidden=min(512, 2 * REPR_DIM)).to(device)
    optimizer = optim.AdamW(mine_test.parameters(), lr=LR)
    mi_critic_logs = []
    for ep in range(EPOCHS):
        logs = train_mine(mine_test, optimizer, params.model, data_loaders['test'], device)
        mi_critic_logs += logs
    test_mi_xz = logs[-1]
    plot_mi_loss_log(mi_critic_logs, "MINE_test_data")
    ## CLUB
    #club_test = CLUB(DATA_DIM, REPR_DIM, min(512, 2 * REPR_DIM)).to(device)
    #optimizer_test = optim.AdamW(club_test.parameters(), lr=LR)
    #mi_critic_logs = []
    #for ep in range(EPOCHS):
    #    logs = train_club(club_test, optimizer_test, params.model, data_loaders['test'], device)
    #    mi_critic_logs += logs
    #test_mi_xz = logs[-1]
    #plot_mi_loss_log(mi_critic_logs, "CLUB_test_data")
    #doe_l_test = DoE(DATA_DIM, REPR_DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)
    #optim_test = optim.AdamW(doe_l_test.parameters(), lr=LR)
    #mi_critic_logs = []
    # for test set it might be also better not to bias
    #for ep in range(1):
    #    L_doe_l_test, logs = train_doe(doe_l_test, optim_test, params.model, data_loaders['test'], device, args.llm_mode)
    #    mi_critic_logs += logs
    #test_mi_xz = report_MI(L_doe_l_test)
    #plot_mi_loss_log(mi_critic_logs, "DoE_test_data")
    print("Full MI test ", test_mi_xz)

    pickle.dump({
        'train_acc': train_hist['acc'][-1],
        'test_acc': test_hist['acc'][-1],
        'train_loss': train_hist['loss'][-1],
        'test_loss': test_hist['loss'][-1],
        'train_IXZ': train_mi_xz,
        'test_IXZ': test_mi_xz,
    }, open(EXP_PATH / "mine_characteristics.pkl", "wb"))
