import argparse
import collections
import matplotlib
from matplotlib import pyplot as plt
from doe_mi_comparison.util import *
import random

BS = 64
LR = 0.01
GRAD_CLIP = 1

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
    def __init__(self, dim, noise):
        self.dim = dim
        self.noise = noise

    def lin_func(self, x):
        return 2 * x + 4.5

    def create_samples(self, num_samples, noise_samples):
        mean = np.zeros(self.dim)
        cov = np.identity(self.dim)
        X = np.random.multivariate_normal(mean, cov, num_samples)
        fx = self.lin_func(X)
        mean_d = np.ones(self.dim)
        cov_d = np.identity(self.dim) * self.noise
        eps = np.random.multivariate_normal(mean_d, cov_d, num_samples * noise_samples)
        p = np.random.permutation(num_samples*noise_samples)
        self.Y = (np.repeat(fx, noise_samples, axis=0) * eps)[p]
        self.X = (np.repeat(X, noise_samples, axis=0))[p]
        return num_samples*noise_samples

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    estimations_doe, test_estimations_doe = collections.OrderedDict(), collections.OrderedDict()
    estimations_doe_l, test_estimations_doe_l = collections.OrderedDict(), collections.OrderedDict()
    for NOISE in [0.01, 0.05, 0.1, 0.3, 0.45]:
        print("noise level", NOISE)

        estimations_doe[NOISE] = []
        estimations_doe_l[NOISE] = []

        pXY = GausDropoutNetworkReprs(args.DIM, NOISE)
        data_size = pXY.create_samples(X_SIZE, NOISE_SAMPLES)

        doe = DoE(args.DIM, args.DIM, critic_params['hidden'], critic_params['layers'], 'gauss').to(device)
        doe_l = DoE(args.DIM, args.DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)

        optim_doe = torch.optim.Adam(doe.parameters(), lr=LR)
        optim_doe_l = torch.optim.Adam(doe_l.parameters(), lr=LR)

        for step in range(1, data_size//BS):
            X = torch.tensor(pXY.X[BS*(step-1):BS*step], dtype=torch.float32).to(device)
            Y = torch.tensor(pXY.Y[BS*(step-1):BS*step], dtype=torch.float32).to(device)
            # doe
            optim_doe.zero_grad()
            L_doe = doe(X, Y)
            L_doe.backward()
            nn.utils.clip_grad_norm_(doe.parameters(), GRAD_CLIP)
            optim_doe.step()
            # doe_l
            optim_doe_l.zero_grad()
            L_doe_l = doe_l(X, Y)
            L_doe_l.backward()
            nn.utils.clip_grad_norm_(doe_l.parameters(), GRAD_CLIP)
            optim_doe_l.step()

            print('step {:4d} | '.format(step), end='')
            print('doe: {:6.2f} | '.format(-L_doe), end='')
            print('doe_l: {:6.2f} | '.format(-L_doe_l))

        # Final evaluation
        X = torch.tensor(pXY.X[BS*(step-1):], dtype=torch.float32).to(device)
        Y = torch.tensor(pXY.Y[BS*(step-1):], dtype=torch.float32).to(device)
        doe.eval()
        test_estimations_doe[NOISE] = -doe(X, Y).item()
        doe_l.eval()
        test_estimations_doe_l[NOISE] = -doe_l(X, Y).item()

    # plotting
    colors = matplotlib.cm.viridis(np.linspace(0.1, 0.9, 4))

    def plot_estimations(est, xlabel, colors, title):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(xlabel)
        # ax1.set_xscale('log')
        ax1.set_ylabel("MI(X, Z)")
        ax1.plot(list(est.keys()), list(est.values()), label="Estimate, 3 samples", color=colors[0])
        ax1.legend(loc='best')
        plt.title(title)
        plt.show()

    plot_estimations(test_estimations_doe, "Noise", colors, "Test estimation doe")
    plot_estimations(test_estimations_doe_l, "Noise", colors, "Test estimation doe_l")
