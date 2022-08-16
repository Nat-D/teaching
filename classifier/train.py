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

def main(num_epoch=10,
         learning_rate=0.001,
         batch_size=32,
         num_workers=1, 
         ):
    
    """
    1. get dataset loader
        1.1 define preprocessing transform steps
        1.2 create Dataset object (define where to load data)
        1.3 create DataLoader object (define batchsize and how to load data)
    """
    
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
    """
    2. model components
       2.1 network
       2.2 loss function
       2.3 optimizer
    """
    network = MyNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # 3. logger object
    logger = Logger(device)

    # 4. training loop
    for epoch in range(num_epoch):
        for batch_idx, (x, target) in enumerate(train_loader):
            
            # 4.0 use GPU if possible
            x = x.to(device)
            target = target.to(device)

            # 4.1 make prediction
            y_pred = network(x)
            
            # 4.2 compute loss 
            loss = loss_fn(y_pred, target)
            
            # 4.3 compute gradients
            optimizer.zero_grad()
            loss.backward()

            # 4.4 adjust the weights
            optimizer.step()

            # 4.5 collect result into the logger
            logger.log_step(loss.item())

        logger.log_epoch(network)


if __name__ == "__main__":
    main()