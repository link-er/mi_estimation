import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import argparse
import collections
import matplotlib
from matplotlib import pyplot as plt
from estimators import estimate_mutual_information
from torch.utils.data import Dataset, DataLoader

from utils import ConcatCritic
from tqdm import *

NOISE = 0.4
PLOT_VALUES = {}

CLIP = None
BS = 64
EPOCHS = 3

parser = argparse.ArgumentParser()
parser.add_argument('--dim', dest='DIM', type=int, help='dimensionality of the input space')
parser.add_argument('--fdim', dest='F_DIM', type=int, help='dimensionality of the representation space')
args = parser.parse_args()

critic_params = {
    'dim': args.DIM,
    'layers': 5, #2,
    'embed_dim': BS//2,
    'hidden_dim': 1024, #256,
    'activation': 'relu',
}
opt_params = {
    'iterations': 10000,
    'learning_rate': 5e-4,
}

if __name__ == "__main__":
    def train_step(critic, opt_crit, data_batch, estimator):
        opt_crit.zero_grad()
        x, y = data_batch
        mi = estimate_mutual_information(estimator, x, y, critic, None, None, clip=CLIP)
        loss = -mi
        loss.backward()
        opt_crit.step()
        return mi

    def test_step(critic, data_batch, estimator):
        x, y = data_batch
        mi = estimate_mutual_information(estimator, x, y, critic, None, None, clip=CLIP)
        return mi

    class VariablesDataset(Dataset):
        def __init__(self, x, fx, fdim, noise_samples, noise_var):
            self.x = np.repeat(x, noise_samples, axis=0)
            mean_d = np.ones(fdim)
            cov_d = np.identity(fdim) * noise_var
            eps = np.random.multivariate_normal(mean_d, cov_d, len(fx) * noise_samples)
            self.z = np.repeat(fx, NOISE_SAMPLES, axis=0) * eps

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return (torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.z[idx], dtype=torch.float32)), 0

    estimations = collections.OrderedDict()
    test_estimations = collections.OrderedDict()
    for NOISE_SAMPLES in [1, 4, 8]:
        print("NOISE_SAMPLES", NOISE_SAMPLES)
        estimations[NOISE_SAMPLES] = collections.OrderedDict()
        test_estimations[NOISE_SAMPLES] = collections.OrderedDict()
        for X_SIZE in [500, 1000, 5000, 10000, 20000, 50000]:
            print("X_SIZE", X_SIZE)
            mean_x = np.zeros(args.DIM)
            cov_x = np.identity(args.DIM)
            x = np.random.multivariate_normal(mean_x, cov_x, X_SIZE)
            fx = 2 * x + 4.5

            training_data = VariablesDataset(x, fx, args.F_DIM, NOISE_SAMPLES, NOISE)
            test_x = np.random.multivariate_normal(mean_x, cov_x, X_SIZE//3)
            test_data = VariablesDataset(test_x, 2 * test_x + 4.5, args.F_DIM, NOISE_SAMPLES, NOISE)
            train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True, drop_last=True)
            test_dataloader = DataLoader(test_data, batch_size=BS, shuffle=True, drop_last=True)

            critic = ConcatCritic(rho=None, **critic_params).cuda()

            opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])
            estimates = []
            for e in tqdm(range(EPOCHS)):
                for batch in train_dataloader:
                    data_batch = batch[0]
                    mi = train_step(critic, opt_crit, data_batch, estimator='smile')
                    mi = mi.detach().cpu().item()
            with torch.no_grad():
                test_mi = 0.0
                c = 0
                for batch in test_dataloader:
                    data_batch = batch[0]
                    test_mi += test_step(critic, data_batch, estimator='smile').detach().cpu().item()
                    c += 1
                test_mi /= c

            estimations[NOISE_SAMPLES][X_SIZE] = mi
            test_estimations[NOISE_SAMPLES][X_SIZE] = test_mi

    colors = matplotlib.cm.viridis(np.linspace(0.1, 0.9, 5))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Number of samples")
    ax1.set_xscale('log')
    ax1.set_ylabel("MI(X, Z)")
    i = 0
    for k in estimations:
        ax1.plot(list(estimations[k].keys()), list(estimations[k].values()), label="Estimate, " + str(k) + "samples",
                 color=colors[i])
        i += 1
    ax1.legend(loc='best')
    plt.title("Train data estimation")
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Number of samples")
    ax1.set_xscale('log')
    ax1.set_ylabel("MI(X, Z)")
    i = 0
    for k in test_estimations:
        ax1.plot(list(test_estimations[k].keys()), list(test_estimations[k].values()), label="Estimate, " + str(k) + "samples",
                 color=colors[i])
        i += 1
    ax1.legend(loc='best')
    plt.title("Test data estimation")
    plt.show()
