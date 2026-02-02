import argparse
import collections
import matplotlib
from matplotlib import pyplot as plt
from club import *
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

    estimations, test_estimations = collections.OrderedDict(), collections.OrderedDict()
    for NOISE in [0.01, 0.05, 0.1, 0.3, 0.45]:
        print("noise level", NOISE)

        estimations[NOISE] = []

        pXY = GausDropoutNetworkReprs(args.DIM, NOISE)
        data_size = pXY.create_samples(X_SIZE, NOISE_SAMPLES)

        model = CLUB(args.DIM, args.DIM, critic_params['hidden']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), LR)

        for step in range(1, data_size//BS):
            X = torch.tensor(pXY.X[BS*(step-1):BS*step], dtype=torch.float32).to(device)
            Y = torch.tensor(pXY.Y[BS*(step-1):BS*step], dtype=torch.float32).to(device)

            model.eval()
            est_mi = model(X, Y).item()
            estimations[NOISE].append(est_mi)
            print('step {:4d} | '.format(step), end='')
            print('club: {:6.2f} | '.format(est_mi))

            model.train()

            model_loss = model.learning_loss(X, Y)

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            del X, Y
            torch.cuda.empty_cache()

        # Final evaluation
        X = torch.tensor(pXY.X[BS*(step-1):], dtype=torch.float32).to(device)
        Y = torch.tensor(pXY.Y[BS*(step-1):], dtype=torch.float32).to(device)
        model.eval()
        test_estimations[NOISE] = model(X, Y).item()

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

    plot_estimations(test_estimations, "Noise", colors, "Test estimation CLUB")
