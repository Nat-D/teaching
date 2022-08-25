import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2


from torch.utils.data import DataLoader

from model import MyNet
from logger import Logger
from dataset import CatDogMiniDataset

import re
import platform

def test_main(num_epoch=50,
         learning_rate=0.001,
         batch_size=32,
         num_workers=1, 
         ):
  
    train_transform = A.Compose([
                        A.Resize(height=32, width=32),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std =[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,
                            ),
                        ToTensorV2()
                        ])

    train_data_object = CatDogMiniDataset(image_dir='data/train/', 
                                          transform=train_transform)


    train_loader = DataLoader(train_data_object, 
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=True
                              )
    network = MyNet()
    
    this_device = platform.platform()
    if torch.cuda.is_available():
        device = "cuda"
    elif re.search("arm64", this_device):
        # use Apple GPU
        device = "mps"
    else:
        device = "cpu"


    network.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    logger = Logger(device)


    x, target = next(iter(train_loader))
    for epoch in range(num_epoch):
        x = x.to(device)
        target = target.to(device)
        y_pred = network(x)
        loss = loss_fn(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == "__main__":
    test_main()