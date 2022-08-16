import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
Important pytorch concepts:
1. nn.Module      - a stateful neural network layer, e.g. nn.Conv2d 
                    (require initialization)
2. nn.Parameter   - parameter or the state variable in a module 
                    (created when a module is initialized)
3. nn.functional  - a stateless function, e.g. F.relu()
                    (can be added directly in a forward method)  
"""

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # initialize the module objects you need in your model
        self.conv1 = nn.Conv2d(in_channels  = 3, 
                               out_channels = 32, 
                               kernel_size  = 3, 
                               stride       = 2)

        self.conv2 = nn.Conv2d(in_channels  = 32, 
                               out_channels = 32, 
                               kernel_size  = 3, 
                               stride       = 2)

        self.linear = nn.Linear(in_features  = 1568,
                                 out_features = 125)

        self.out = nn.Linear(in_features  = 125,
                                 out_features = 2)
        

    def forward(self, x):
        
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.reshape(h.shape[0], -1)
        h = F.relu(self.linear(h))
        y = self.out(h)

        return y

        