import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import ortho_group


class GausDropoutNetworkReprs(Dataset):
    def __init__(self, dim, noise, num_samples, noise_samples):
        """ 
        dim: dimensionality 
        noise: variance scaling for multiplicative noise 
        num_samples: number of X 
        noise_samples: nb of positive samples per X 
        total number of samples given is: num_samples * noise_samples 
        """
        super().__init__()
        self.dim = dim
        self.noise = noise
        self.num_samples = num_samples
        self.noise_samples = noise_samples
        self._create_samples()

    def lin_func(self, x):
        return 2 * x + 4.5

    def _create_samples(self):
        mean = np.zeros(self.dim)
        cov = np.identity(self.dim)

        X = np.random.multivariate_normal(mean, cov, self.num_samples)
        fx = self.lin_func(X)

        mean_d = np.ones(self.dim)
        cov_d = np.identity(self.dim) * self.noise
        eps = np.random.multivariate_normal(
            mean_d, cov_d, self.num_samples * self.noise_samples
        )

        X_rep = np.repeat(X, self.noise_samples, axis=0)
        fx_rep = np.repeat(fx, self.noise_samples, axis=0)

        p = np.random.permutation(self.num_samples * self.noise_samples)

        Y = (fx_rep * eps)[p]
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
    Y can optionally be embedded into higher dimension via orthogonal projection.
    """

    def __init__(self, dim, noise, num_samples, add_dim=0):
        super().__init__()

        self.dim = dim
        self.noise = noise
        self.num_samples = num_samples
        self.add_dim = add_dim

        self._create_samples()

    def lin_func(self, x):
        return 2 * x + 4.5

    def _create_samples(self):
        mean = np.zeros(self.dim)
        cov = np.identity(self.dim)

        X = np.random.multivariate_normal(mean, cov, self.num_samples)
        fx = self.lin_func(X)

        mean_d = np.ones(self.dim)
        cov_d = np.identity(self.dim) * self.noise
        eps = np.random.multivariate_normal(mean_d, cov_d, self.num_samples)

        Y = fx * eps

        if self.add_dim > 0:
            transform = ortho_group.rvs(self.dim + self.add_dim)
            Y = Y @ transform[:self.dim, :]

        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class GausDropoutHeterosc(Dataset):
    """
    Gaussian X with equicorrelated covariance:

        Sigma_ii = 1
        Sigma_ij = cov_value  (i ≠ j)

    Y = (2X + 4.5) * multiplicative Gaussian noise
    eps ~ N(1, noise * I)
    """

    def __init__(self, dim, noise, num_samples, cov_value):
        super().__init__()

        self.dim = dim
        self.noise = noise
        self.num_samples = num_samples
        self.cov_value = cov_value

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

        # Multiplicative Gaussian noise
        eps = np.random.multivariate_normal(
            np.ones(self.dim),
            np.eye(self.dim) * self.noise,
            self.num_samples
        )

        Y = fx * eps

        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
