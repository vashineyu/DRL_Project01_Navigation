# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections.abc import Iterable


class MLPArchitecture(nn.Module):
    def __init__(self, batch_size, n_outputs, state_size):
        super(MLPArchitecture, self).__init__()
        if isinstance(state_size, Iterable):
            assert len(state_size)==1
            state_size = state_size[0]
        self.batch_size = batch_size
        self.n_outputs = n_outputs
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_size, 128) 
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, n_outputs)        
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        out = self.fc3(h)
        return out

    
class CNNArchitecture(nn.Module):
    def __init__(self, batch_size, n_outputs, state_size):
        super(CNNArchitecture, self).__init__()
        self.batch_size = batch_size
        self.n_outputs = n_outputs
        
        self.conv1 = nn.Conv2d(state_size[-1], 32, kernel_size=5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64*7*7, 50)
        self.fc2 = nn.Linear(50, n_outputs)
        self.relu = nn.ReLU()    
    
    def forward(self, x):
        x = x.permute(0,3,1,2) # fix pytorch format 0 1 2 3 4 5 6 7 8 9
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        
        h = h.view(-1, 64*7*7)
        
        h = self.relu(self.fc1(h))
        out = self.relu(self.fc2(h))
        return out

class NNModel():
    def __init__(self, arch, batch_size, n_outputs, state_shape, learning_rate=1e-3):
        self.net=arch(batch_size, n_outputs, state_shape)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
    def get_loss(self, y, y_hat):
        return nn.MSELoss()(y.detach(), y_hat).mean()
    
    def train(self, batch_x, batch_y, actions):
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(batch_y).float()
        actions = torch.from_numpy(actions).long()
        y_hat = self.net.forward(batch_x)
        loss = self.get_loss(batch_y, y_hat.gather(1, actions))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
    def predict(self, batch_x, is_tensor=False):
        if not is_tensor:
            batch_x = torch.from_numpy(batch_x).float()
        prediction = self.net.forward(batch_x)
        return prediction.detach().numpy()
    
    def copy_weights_from(self, net, tau=0.001):
        # tau should be a small parameter
        for local_param, ext_param in zip(self.net.parameters(), net.parameters()):
            local_param.data.copy_((1-tau)*(local_param.data) + (tau)*ext_param.data)