import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import argparse
import collections
import matplotlib
from matplotlib import pyplot as plt
from mi_estimator import estimate_mutual_information
from torch.utils.data import Dataset, DataLoader

from mi_estimators_local.dropout_MI import gaussian_noise_mi

from critics import ConcatCritic
from tqdm import *

CLIP = 5 #None
BS = 256
LR = 1e-3
EPOCHS = 25
X_SIZE = 20000
TEST_SIZE = 5000

smooth_win = 500

parser = argparse.ArgumentParser()
parser.add_argument('--dim', dest='DIM', type=int, help='dimensionality of the input space')
parser.add_argument('--fdim', dest='F_DIM', type=int, help='dimensionality of the representation space')
args = parser.parse_args()

critic_params = {
    'dim': args.DIM,
    'layers': 5, #2,
    'hidden_dim': 1024, #256,
    'activation': 'relu',
}

def train_step(critic, opt_crit, scheduler, data_batch):
    opt_crit.zero_grad()
    x, y = data_batch
    # TODO: is it normal that it can return negative value? seems like regularizing for it helps training
    mi = estimate_mutual_information(x, y, critic, clip=CLIP)
    loss = -mi
    loss.backward()
    opt_crit.step()
    scheduler.step()
    return mi

def test_step(critic, data_batch):
    x, y = data_batch
    mi = estimate_mutual_information(x, y, critic, clip=CLIP)
    return mi

class MultiplicativeDataset(Dataset):
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

def lin_func(x):
    return 2 * x + 4.5

if __name__ == "__main__":
    estimations = collections.OrderedDict()
    test_estimations = collections.OrderedDict()
    colors_train = matplotlib.cm.viridis(np.linspace(0.1, 0.9, 4*4))
    i = 0
    for NOISE_SAMPLES in [3]: #[1, 2, 4, 8]:
        #print("noise samples", NOISE_SAMPLES)
        #estimations[NOISE_SAMPLES] = collections.OrderedDict()
        #test_estimations[NOISE_SAMPLES] = collections.OrderedDict()
        for NOISE in [0.01, 0.05, 0.1, 0.3]:
            print("noise level", NOISE)
            mean_x = np.zeros(args.DIM)
            cov_x = np.identity(args.DIM)
            x = np.random.multivariate_normal(mean_x, cov_x, X_SIZE)
            fx = lin_func(x)

            #training_data = MultiplicativeDataset(x, fx, args.F_DIM, NOISE_SAMPLES, NOISE)
            #train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True, drop_last=True)

            test_x = np.random.multivariate_normal(mean_x, cov_x, TEST_SIZE)
            '''
            test_data = MultiplicativeDataset(test_x, lin_func(test_x), args.F_DIM, NOISE_SAMPLES, NOISE)
            test_dataloader = DataLoader(test_data, batch_size=BS, shuffle=True, drop_last=True)

            critic = ConcatCritic(**critic_params).cuda()

            opt_crit = optim.Adam(critic.parameters(), lr=LR)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_crit, EPOCHS)
            estimates = []
            smoothed_est = []
            keep_train = True
            total_ind = 0
            for e in tqdm(range(EPOCHS)):
                if not keep_train:
                    break
                for batch in train_dataloader:
                    total_ind += 1
                    data_batch = batch[0]
                    mi = train_step(critic, opt_crit, scheduler, data_batch)
                    mi = mi.detach().cpu().item()
                    estimates.append(mi)
                    if total_ind > smooth_win:
                        smoothed_est.append(np.mean(estimates[-smooth_win:]))
                        #early stopping criterion based on not changing values
                        #TODO: try 50/noise, it seems that with smaller noise it is more unstable
                        #TODO: it should also depend on dimension and variance
                        if len(smoothed_est) > smooth_win and abs(smoothed_est[-smooth_win] - smoothed_est[-1]) < 0.01:
                            print("No changes for long time, breaking training")
                            keep_train = False
                            break

            plt.plot(list(range(len(estimates))), estimates, alpha=0.6, c=colors_train[i])
            plt.plot(np.array(range(len(smoothed_est)))+smooth_win-1, smoothed_est, c=colors_train[i])
            i += 1
            #plt.ylim(-0.1, 220)
            plt.title('Critic training')
            plt.savefig("training_critic_noise" + str(NOISE) + "_samples" + str(NOISE_SAMPLES) + ".jpg")
            plt.close()

            with torch.no_grad():
                test_mi = 0.0
                c = 0
                for batch in test_dataloader:
                    data_batch = batch[0]
                    test_mi += test_step(critic, data_batch).detach().cpu().item()
                    c += 1
                test_mi /= c

            estimations[NOISE] = np.array(estimates[-500:]).mean()
            test_estimations[NOISE] = test_mi
            '''

            estimations[NOISE] = gaussian_noise_mi(fx, NOISE, sampling=3)
            test_estimations[NOISE] = gaussian_noise_mi(test_x, NOISE, sampling=3)

    # plotting
    colors = matplotlib.cm.viridis(np.linspace(0.1, 0.9, 4))

    def plot_estimations(est, xlabel, colors, title):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(xlabel)
        # ax1.set_xscale('log')
        ax1.set_ylabel("MI(X, Z)")
        #i = 0
        #for k in est:
        ax1.plot(list(est.keys()), list(est.values()), label="Estimate, 3 samples", color=colors[0])
        #    i += 1
        ax1.legend(loc='best')
        plt.title(title)
        plt.show()

    plot_estimations(estimations, "Noise", colors, "Train estimation")
    plot_estimations(test_estimations, "Noise", colors, "Test estimation")
