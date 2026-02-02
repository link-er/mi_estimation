import torch
import torch.nn as nn
import torch.nn.functional as F

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden=256, ema_decay=0.99):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.ema_decay = ema_decay
        self.register_buffer("ema_term", torch.tensor(1.0))

    def forward(self, x, y):
        joint = torch.cat([x, y], dim=1)
        t = self.net(joint)

        perm = torch.randperm(y.size(0))
        y_shuffle = y[perm]
        marg = torch.cat([x, y_shuffle], dim=1)
        t_marg = self.net(marg)

        # optional stability clamp
        t_marg_clamped = torch.clamp(t_marg, max=20)

        et = torch.exp(t_marg_clamped)

        with torch.no_grad():
            new_ema = self.ema_term * self.ema_decay + (1 - self.ema_decay) * et.mean()
            self.ema_term.copy_(new_ema)

        mi = t.mean() - torch.log(self.ema_term + 1e-8)

        return mi, t.mean().detach(), et.mean().detach()