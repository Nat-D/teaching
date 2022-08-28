import torch
import numpy as np

from dataset import CatDogMiniDataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

class Logger():
    def __init__(self, device):
        
        self.device = device
        self.training_loss = 0 
        self.training_step = 0

        val_transform = A.Compose([
                        A.Resize(height=32, width=32),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std =[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,
                            ),
                        ToTensorV2()
                        ])

        val_data_object = CatDogMiniDataset(
                            image_dir='data/val/',
                            transform=val_transform
                            )

        self.dataloader = DataLoader(val_data_object,
                            batch_size=84,
                            num_workers=1,
                            pin_memory=True,
                            shuffle=False
                            )

    def log_step(self, loss):
        if self.training_step % 20 == 0:
            print(f"Training at step: {self.training_step}, Loss: {loss}")
        
        self.training_loss += loss
        self.training_step += 1


    def log_epoch(self, network):

        # Training Set
        print(f"(Train Set) Avg Loss: {self.training_loss / self.training_step}")
        self.training_loss = 0 

        # Validation Set
        num_correct = 0
        num_samples = 0
        network.eval()

        with torch.no_grad():
            for x, targets in self.dataloader:
                x = x.to(self.device)
                targets = targets.to(self.device)

                scores = network(x)
                _, predictions = scores.max(1)

                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

            print( 
                f"(Val Set) Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
                )

        network.train()
