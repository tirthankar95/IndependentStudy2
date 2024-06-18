import torch 
import torch.optim as optim 
import torch.nn as nn 
from torch.distributions import Categorical

class NNet(nn.Module):
    def __get_dim__(self, dummyX, nn_list):
        for nn_arch in nn_list:
           dummyX = nn_arch(dummyX)
        return dummyX.shape 
    def __init__(self, dummyX, noOfActions: int):
        super().__init__()
        self.scale = 1
        self.op_shape = []
        self.base_cnn = nn.Sequential(
            # Input channels = 1, output channels = 8, kernel size = 3
            nn.Conv2d(1, 8*self.scale, 3), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
            )
        self.op_shape.append(self.__get_dim__(dummyX, [self.base_cnn]))
        self.base = nn.Sequential(
            nn.Linear(self.op_shape[-1][-1], 32*self.scale), 
            nn.ReLU()
        )
        self.op_shape.append(self.__get_dim__(dummyX, [self.base_cnn, self.base]))
        self.actor = nn.Sequential(
            nn.Linear(self.op_shape[-1][-1], 16*self.scale),
            nn.ReLU(),
            nn.Linear(16*self.scale, noOfActions)
            ) 
        self.critic = nn.Sequential(
            nn.Linear(self.op_shape[-1][-1], 16*self.scale),
            nn.ReLU(),
            nn.Linear(16*self.scale,1)
            )    
    def forward(self, x):
        x = self.base_cnn(x)
        x = self.base(x)
        mu_dist = Categorical(logits = self.actor(x))
        value = self.critic(x)
        return mu_dist, value
    