import argparse
import collections
import matplotlib
from matplotlib import pyplot as plt
from doe_mi_comparison.util import *
import random
from scipy.stats import ortho_group

BS = 64
LR = 0.01
GRAD_CLIP = 1
NOISE = 0.3

X_SIZE = 50000
NOISE_SAMPLES = 3

parser = argparse.ArgumentParser()
parser.add_argument('--dim', dest='DIM', type=int, help='dimensionality of the input space')
args = parser.parse_args()

critic_params = {
    'layers': 1,
    'hidden': 128
}

class GausDropoutNetworkReprs(object):
    def __init__(self, dim, add_dim):
        self.dim = dim
        self.add_dim = add_dim

    def lin_func(self, x):
        return 2*x + 4.5

    def create_samples(self, num_samples, noise_samples):
        mean = np.zeros(self.dim)
        cov = np.identity(self.dim)
        X = np.random.multivariate_normal(mean, cov, num_samples)
        # apply linear transformation
        fx = self.lin_func(X)
        # generate noise in the amount according to noise_samples
        mean_d = np.ones(self.dim)
        cov_d = np.identity(self.dim) * NOISE
        eps = np.random.multivariate_normal(mean_d, cov_d, num_samples * noise_samples)
        # permute samples for easy batching
        p = np.random.permutation(num_samples*noise_samples)
        self.Y = (np.repeat(fx, noise_samples, axis=0) * eps)[p]
        self.X = (np.repeat(X, noise_samples, axis=0))[p]
        transform = ortho_group.rvs(self.dim + self.add_dim)
        self.Y_emb = self.Y @ np.array(transform[:self.dim, :])
        return num_samples*noise_samples

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    estimations_doe, test_estimations_doe = collections.OrderedDict(), collections.OrderedDict()
    estimations_doe_l, test_estimations_doe_l = collections.OrderedDict(), collections.OrderedDict()
    for ADDDIM in [5, 10, 20, 30]:
        print("added dimensionality", ADDDIM)

        estimations_doe[ADDDIM] = []
        estimations_doe_l[ADDDIM] = []

        pXY = GausDropoutNetworkReprs(args.DIM, ADDDIM)
        data_size = pXY.create_samples(X_SIZE, NOISE_SAMPLES)

        doe = DoE(args.DIM, args.DIM, critic_params['hidden'], critic_params['layers'], 'gauss').to(device)
        doe_l = DoE(args.DIM, args.DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)
        doe_addim = DoE(args.DIM, args.DIM + ADDDIM, critic_params['hidden'], critic_params['layers'], 'gauss').to(device)
        doe_l_addim = DoE(args.DIM, args.DIM + ADDDIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)

        optim_doe = torch.optim.Adam(doe.parameters(), lr=LR)
        optim_doe_l = torch.optim.Adam(doe_l.parameters(), lr=LR)
        optim_doe_addim = torch.optim.Adam(doe_addim.parameters(), lr=LR)
        optim_doe_l_addim = torch.optim.Adam(doe_l_addim.parameters(), lr=LR)

        for step in range(1, data_size//BS):
            X = torch.tensor(pXY.X[BS*(step-1):BS*step], dtype=torch.float32).to(device)
            Y = torch.tensor(pXY.Y[BS*(step-1):BS*step], dtype=torch.float32).to(device)
            Y_emb = torch.tensor(pXY.Y_emb[BS*(step-1):BS*step], dtype=torch.float32).to(device)
            # doe
            optim_doe.zero_grad()
            L_doe = doe(X, Y)
            L_doe.backward()
            nn.utils.clip_grad_norm_(doe.parameters(), GRAD_CLIP)
            optim_doe.step()
            optim_doe_addim.zero_grad()
            L_doe_addim = doe_addim(X, Y_emb)
            L_doe_addim.backward()
            nn.utils.clip_grad_norm_(doe_addim.parameters(), GRAD_CLIP)
            optim_doe_addim.step()
            # doe_l
            optim_doe_l.zero_grad()
            L_doe_l = doe_l(X, Y)
            L_doe_l.backward()
            nn.utils.clip_grad_norm_(doe_l.parameters(), GRAD_CLIP)
            optim_doe_l.step()
            optim_doe_l_addim.zero_grad()
            L_doe_l_addim = doe_l_addim(X, Y_emb)
            L_doe_l_addim.backward()
            nn.utils.clip_grad_norm_(doe_l_addim.parameters(), GRAD_CLIP)
            optim_doe_l_addim.step()

            print('step {:4d} | '.format(step), end='')
            print('doe: {:6.2f} | '.format(-L_doe), end='')
            print('doe_l: {:6.2f} | '.format(-L_doe_l), end='')
            print('doe with addition: {:6.2f} | '.format(-L_doe_addim), end='')
            print('doe_l with addition: {:6.2f} | '.format(-L_doe_l_addim))

        # Final evaluation
        X = torch.tensor(pXY.X[BS*(step-1):], dtype=torch.float32).to(device)
        Y = torch.tensor(pXY.Y[BS*(step-1):], dtype=torch.float32).to(device)
        Y_emb = torch.tensor(pXY.Y_emb[BS*(step-1):], dtype=torch.float32).to(device)
        doe.eval()
        doe_addim.eval()
        test_estimations_doe[ADDDIM] = doe(X, Y).item() / doe_addim(X, Y_emb).item()
        doe_l.eval()
        doe_l_addim.eval()
        test_estimations_doe_l[ADDDIM] = doe_l(X, Y).item() / doe_l_addim(X, Y_emb).item()

    # plotting
    colors = matplotlib.cm.viridis(np.linspace(0.1, 0.9, 4))

    def plot_estimations(est, xlabel, colors, title):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(xlabel)
        # ax1.set_xscale('log')
        ax1.set_ylabel("MI(X, Z)")
        ax1.plot(list(est.keys()), list(est.values()), label="Estimate", color=colors[0])
        ax1.legend(loc='best')
        plt.title(title)
        plt.show()

    plot_estimations(test_estimations_doe, "Embedding", colors, "Test estimation doe")
    plot_estimations(test_estimations_doe_l, "Embedding", colors, "Test estimation doe_l")
