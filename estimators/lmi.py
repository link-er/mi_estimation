import torch 
import torch.nn as nn
import torch.nn.functional as F

from lmi_utils import Encoder, Decoder


class LMI(nn.Module):


    def __init__(self, x_dim, y_dim, hidden_dim):
        super.__init__()
        enc_x = Encoder(x_dim, hidden_dim)
        enc_y = Encoder(y_dim, hidden_dim)

        dec_xx = Decoder(x_dim, hidden_dim)
        dec_yy = Decoder(y_dim, hidden_dim)
        dec_xy = Decoder(y_dim, hidden_dim)
        dec_yx = Decoder(x_dim, hidden_dim)


    def forward(self, x, y):
        zx, zy = self.enc_x(x), self.enc_y(y)

        ae_loss = F.mse_loss(x, self.dec_xx(zx)) + F.mse_loss(y, self.dec_yy(zy))
        cl_loss = F.mse_loss(x, self.dec_yx(zy)) + F.mse_loss(y, self.dec_xy(zx))

