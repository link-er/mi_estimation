import argparse
import collections
import matplotlib
from matplotlib import pyplot as plt
from club import *
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

    estimations, test_estimations = collections.OrderedDict(), collections.OrderedDict()
    for ADDDIM in [5, 10, 20, 30]:
        print("added dimensionality", ADDDIM)

        estimations[ADDDIM] = []

        pXY = GausDropoutNetworkReprs(args.DIM, ADDDIM)
        data_size = pXY.create_samples(X_SIZE, NOISE_SAMPLES)

        model = CLUB(args.DIM, args.DIM, critic_params['hidden']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), LR)

        model_addim = CLUB(args.DIM, args.DIM + ADDDIM, critic_params['hidden']).to(device)
        optimizer_addim = torch.optim.Adam(model_addim.parameters(), LR)

        for step in range(1, data_size // BS):
            X = torch.tensor(pXY.X[BS * (step - 1):BS * step], dtype=torch.float32).to(device)
            Y = torch.tensor(pXY.Y[BS * (step - 1):BS * step], dtype=torch.float32).to(device)
            Y_emb = torch.tensor(pXY.Y_emb[BS*(step-1):BS*step], dtype=torch.float32).to(device)

            model.eval()
            est_mi = model(X, Y).item()
            estimations[ADDDIM].append(est_mi)
            print('step {:4d} | '.format(step), end='')
            print('club: {:6.2f} | '.format(est_mi), end='')

            model_addim.eval()
            est_mi_addim = model_addim(X, Y_emb).item()
            estimations[ADDDIM].append(est_mi_addim)
            print('club embedded: {:6.2f} | '.format(est_mi_addim))

            model.train()

            model_loss = model.learning_loss(X, Y)

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            model_addim.train()

            model_loss_addim = model_addim.learning_loss(X, Y_emb)

            optimizer_addim.zero_grad()
            model_loss_addim.backward()
            optimizer_addim.step()

            del X, Y, Y_emb
            torch.cuda.empty_cache()

        # Final evaluation
        X = torch.tensor(pXY.X[BS*(step-1):], dtype=torch.float32).to(device)
        Y = torch.tensor(pXY.Y[BS*(step-1):], dtype=torch.float32).to(device)
        Y_emb = torch.tensor(pXY.Y_emb[BS*(step-1):], dtype=torch.float32).to(device)
        model.eval()
        model_addim.eval()
        test_estimations[ADDDIM] = model(X, Y).item() / model_addim(X, Y_emb).item()

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

    plot_estimations(test_estimations, "Embedding", colors, "Test estimation CLUB")
