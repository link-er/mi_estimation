import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import ortho_group

# you can get a mixed dataset of noisy ys or just pass noise_samples=1 and then get dataset of representations
# you can use droppedout then to modify batch of representations into noisy versions, kind of one sample from dropout
class GausDropoutNetworkReprs(Dataset):
    def __init__(self, dimX, dimY, A, B, noise, num_samples, noise_samples=1):
        """ 
        dimX: dimensionality of X
        dimY: dimensionality of y
        noise: variance scaling for multiplicative noise 
        num_samples: number of X 
        noise_samples: nb of positive samples per X 
        total number of samples given is: num_samples * noise_samples 
        """
        super().__init__()
        self.dimX = dimX
        self.dimY = dimY
        self.A = A
        self.B = B
        self.num_samples = num_samples
        self.noise_samples = noise_samples

        self.noise_distr = (np.ones(self.dimY), np.identity(self.dimY) * noise)

        self._create_samples()

    def perceptron_func(self, x):
        return torch.tanh(torch.matmul(torch.from_numpy(x).float(), self.A).squeeze() + self.B).numpy()

    def droppedout(self, y):
        drp = []
        for yi in y:
            eps = np.random.multivariate_normal(
                self.noise_distr[0], self.noise_distr[1], len(yi)
            )
            drp.append(y*eps)
        return np.array(drp)

    def _create_samples(self):
        mean = np.zeros(self.dimX)
        cov = np.identity(self.dimX)

        X = np.random.multivariate_normal(mean, cov, self.num_samples)
        Y = self.perceptron_func(X)

        eps = np.random.multivariate_normal(
            self.noise_distr[0], self.noise_distr[1], self.num_samples * self.noise_samples
        )

        X_rep = np.repeat(X, self.noise_samples, axis=0)
        Y_rep = np.repeat(Y, self.noise_samples, axis=0)

        p = np.random.permutation(self.num_samples * self.noise_samples)

        Y = (Y_rep * eps)[p]
        X_rep = X_rep[p]

        self.X = torch.from_numpy(X_rep).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.num_samples * self.noise_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class GausDropoutEmbedded(Dataset):
    """
    Gaussian X
    Y = (2X + 4.5) * multiplicative Gaussian noise
    Y_emb = Y embedded into higher dimension via orthogonal projection
    """

    def __init__(self, dim, noise, num_samples, noise_samples, add_dim=0):
        super().__init__()

        self.dim = dim
        self.noise = noise
        self.num_samples = num_samples
        self.add_dim = add_dim
        self.noise_samples = noise_samples

        self._create_samples()

    def lin_func(self, x):
        return 2 * x + 4.5

    def _create_samples(self):
        mean = np.zeros(self.dim)
        cov = np.identity(self.dim)

        X = np.random.multivariate_normal(mean, cov, self.num_samples)
        fx = self.lin_func(X)

        X_rep = np.repeat(X, self.noise_samples, axis=0)
        fx_rep = np.repeat(fx, self.noise_samples, axis=0)

        # ---- multiplicative noise ----
        mean_d = np.ones(self.dim)
        cov_d = np.identity(self.dim) * self.noise
        eps = np.random.multivariate_normal(
            mean_d,
            cov_d,
            self.num_samples * self.noise_samples
        )

        # ---- generate Y ----
        Y = fx_rep * eps

        # ---- shuffle (important for batching) ----
        p = np.random.permutation(len(Y))
        X_rep = X_rep[p]
        Y = Y[p]

        # ---- embedding ----
        if self.add_dim > 0:
            transform = ortho_group.rvs(self.dim + self.add_dim)
            Y_emb = Y @ transform[:self.dim, :]
        else:
            Y_emb = Y

        # ---- convert to torch ----
        self.X = torch.from_numpy(X_rep).float()
        self.Y = torch.from_numpy(Y).float()
        self.Y_emb = torch.from_numpy(Y_emb).float()


    def __len__(self):
        return self.num_samples * self.noise_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Y_emb[idx]


class GausDropoutHeterosc(Dataset):
    """
    Gaussian X with equicorrelated covariance:

        Sigma_ii = 1
        Sigma_ij = cov_value  (i ≠ j)

    Y = (2X + 4.5) * multiplicative Gaussian noise
    eps ~ N(1, noise * I)
    """

    def __init__(self, dim, noise, num_samples, noise_samples, cov_value):
        super().__init__()

        self.dim = dim
        self.noise = noise
        self.num_samples = num_samples
        self.cov_value = cov_value
        self.noise_samples = noise_samples

        self._create_samples()

    def lin_func(self, x):
        return 2 * x + 4.5

    def _create_samples(self):
        mean = np.zeros(self.dim)

        # Equicorrelated covariance matrix
        cov = np.full((self.dim, self.dim), self.cov_value)
        np.fill_diagonal(cov, 1.0)

        # Sample X
        X = np.random.multivariate_normal(
            mean, cov, self.num_samples
        )

        fx = self.lin_func(X)

        X_rep = np.repeat(X, self.noise_samples, axis=0)
        fx_rep = np.repeat(fx, self.noise_samples, axis=0)

        # Multiplicative Gaussian noise
        eps = np.random.multivariate_normal(
            np.ones(self.dim),
            np.eye(self.dim) * self.noise,
            self.num_samples * self.noise_samples
        )

        Y = fx_rep * eps

        p = np.random.permutation(len(Y))
        X_rep = X_rep[p]
        Y = Y[p]

        self.X = torch.from_numpy(X_rep).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.num_samples  

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
