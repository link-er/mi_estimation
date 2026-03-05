import torch 
import torch.nn as nn

class Encoder(nn.Module):
    '''
    
    '''
    def __init__(self, dim, hidden_dim):
        super.__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, input):
        return self.encoder(input)
    
    

class Decoder(nn.Module):

    def __init__(self, dim, hidden_dim):
        super.__init__()

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, input):
        return self.decoder(input)
    
