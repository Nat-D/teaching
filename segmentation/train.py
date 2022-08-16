import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from model import UNET
from dataset import KittiSegmentMini
from logger import TensorboardLogger

import albumentations as A
from albumentations.pytorch import ToTensorV2


def main(num_epoch=1000,
         learning_rate=0.0001,
         batch_size=8,
         num_workers=1,
         ):

    # 1. get dataset loader
    # 1.1 transform / augmentation
    train_transform = A.Compose([
                        A.Resize(height=32, width=128),
                        A.Rotate(limit=35, p=0.8),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std =[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,
                            ),
                        ToTensorV2()
                        ])
   
    # 1.2 create dataset object
    train_data_object = KittiSegmentMini(image_dir='data/train/image/',
                                         mask_dir='data/train/mask/',
                                         transform = train_transform)

   
    # 1.3 create dataloader
    train_loader = DataLoader(train_data_object,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=True)

    # 2. initiate model / loss function/ optimizer
    network = UNET(in_channels=3, out_channels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # 3. initiate logger
    logger = TensorboardLogger(device)

    # 4. training loop 
    for epoch in range(num_epoch):
        for batch_idx, (image, mask) in enumerate(train_loader):

            image = image.to(device)
            mask = mask.long().to(device)

            # 4.1 make prediction 
            mask_prediction = network(image)
            
            # 4.2 compute loss
            loss = loss_fn(mask_prediction, mask)

            # 4.3 compute gradient
            optimizer.zero_grad()
            loss.backward()

            # 4.4 update weights
            optimizer.step()

            logger.log_step(loss.item())

        logger.log_epoch(network)


if __name__ == "__main__":
    main()


