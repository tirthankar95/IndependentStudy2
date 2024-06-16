import torch 
import torch.optim as optim 
import torch.nn as nn 

class NNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.base = nn.Sequential( nn.Linear(),
                                 nn.ReLU(),
                                 nn.Linear(), 
                                 nn.ReLU()

        )
        self.actor = None 
        self.critic = None 
    def forward():
        pass 
    