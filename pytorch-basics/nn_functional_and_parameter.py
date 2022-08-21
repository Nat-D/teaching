import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F 

# Example of how to create a module with 
# nn.parameter.Parameter and nn.functional
# This class should be the same as nn.Linear (except initialization)
class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnected, self).__init__()

        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = Parameter(torch.empty(out_features))

    def forward(self, x):
        # y = xA^T + b
        # F.linear functional is implemented in c
        return F.linear(x, self.weight, self.bias) 


if __name__ == "__main__":

    fc = FullyConnected(in_features=15, out_features=2)

    for param in fc.parameters():
        print(param.shape)

    x = torch.rand((5, 15), device="cpu") #[batch, in_features]
    y = fc(x)
    print(y)

    