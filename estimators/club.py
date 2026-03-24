import torch.nn as nn

class CLUB(nn.Module):
    """
    CLUB: Mutual Information Contrastive Learning Upper Bound
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]

    forward() returns:
        loss, ub_estimate (MI)
    """

    def __init__(self, x_dim, y_dim, hidden_dim):
        super().__init__()

        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, y_dim),
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, y_dim),
            nn.Tanh(),
        )

    def forward(self, x_samples, y_samples):

        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        var = logvar.exp()

        # -------------------------------------------------
        # Shared squared difference for positive pairs
        # -------------------------------------------------

        diff_pos = mu - y_samples
        sq_diff_pos = diff_pos ** 2

        # -------------------------------------------------
        # Upper Bound Estimate
        # main CLUB formula computation
        # -------------------------------------------------

        positive = - sq_diff_pos / (2.0 * var)

        prediction = mu.unsqueeze(1)         # [B,1,D]
        y_expand = y_samples.unsqueeze(0)    # [1,B,D]

        sq_diff_neg = (y_expand - prediction) ** 2
        negative = - sq_diff_neg.mean(dim=1) / (2.0 * var)

        ub_estimate = (
            positive.sum(dim=-1)
            - negative.sum(dim=-1)
        ).mean()

        # -------------------------------------------------
        # Training Loss (reuse sq_diff_pos & var)
        #  log-likelihood of a multivariate Gaussian N(y∣μ(x),diag(σ2(x)))
        #  summed over dimensions and averaged over samples
        # -------------------------------------------------
        loglikeli = (
            - sq_diff_pos / var
            - logvar
        ).sum(dim=1).mean()

        loss = -loglikeli

        return loss, ub_estimate
