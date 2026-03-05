import torch 

import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    '''
    InfoNCE-based MI lower-bound estimator
    '''
    def __init__(self, x_dim, y_dim, hidden_dim, temperature):
        super().__init__()

        self.enc_x = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.enc_y = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.temperature = temperature


    def forward(self, x_samples, y_samples):
        
        B = x_samples.size(0)
        device = x_samples.device

        zx = self.enc_x(x_samples)
        zy = self.enc_y(y_samples)

        zx = F.normalize(zx, dim=1)
        zy = F.normalize(zy, dim=1)

        logits = torch.matmul(zx, zy.T)/self.temperature

        labels = torch.arange(B, device=device)

        loss = F.cross_entropy(logits, labels)

        lb_estimate = torch.log(torch.tensor(B, device=device, dtype=loss.dtype)) - loss

        return loss, lb_estimate



