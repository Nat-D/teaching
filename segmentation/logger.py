from torch.utils.tensorboard import SummaryWriter
from dataset import KittiSegmentMini

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


color_group = np.array([[0,0,0],  # black - other
                       [255,0,0], # red - road 
                       [0, 255, 0],  # green - vehicle
                       ],dtype=np.uint8)


class TensorboardLogger():
    def __init__(self, device, log_dir='runs'):
        self.device = device

        self.writer = SummaryWriter(log_dir)

        self.training_loss = 0 
        self.training_step = 0

        self.num_steps_per_epoch = 0


        # validation loader
        transform = A.Compose([
                        A.Resize(height=32, width=128),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std =[1.0, 1.0, 1.0],
                            max_pixel_value=255.0,
                            ),
                        ToTensorV2()
                        ])
        data_object = KittiSegmentMini(image_dir='data/val/image/',
                                       mask_dir='data/val/mask/',
                                       transform = transform)

        self.loader = DataLoader(data_object,
                                batch_size=32,
                                num_workers=1,
                                pin_memory=True,
                                shuffle=False)

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def log_step(self, loss):
        self.training_loss += loss
        self.training_step += 1
        self.num_steps_per_epoch += 1


    def log_epoch(self, network):
        
        # Training
        self.writer.add_scalar('Training/Average_trainig_loss',
                                self.training_loss / self.num_steps_per_epoch,
                                self.training_step
                                )
        self.training_loss = 0
        self.num_steps_per_epoch = 0


        # Validation
        self.validate(network)

        # visualize prediction
        self.save_predicitons_as_img(network)


    def validate(self, network):

        network.eval()

        total_loss = 0
        num_step = 0

        total_recall_for_each_class = 0
        total_precision_for_each_class = 0

        with torch.no_grad():
            for x,y in self.loader:
                x = x.to(self.device)
                y = y.long().to(self.device)

                preds = network(x)
                loss = self.loss_fn(preds, y)

                total_loss += loss
                num_step += 1

                preds_max = torch.argmax(preds, dim=1, keepdim=False) # [B,C,H,W] -> [B, H, W]
                preds_max = F.one_hot(preds_max, num_classes=3) # [B, H, W] -> [B,H,W,C]
                y = F.one_hot(y, num_classes=3)

                ones = torch.ones_like(preds_max)
                zeros = torch.zeros_like(preds_max)

                true_pos = torch.logical_and((preds_max == ones), (preds_max == y)) 
                true_pos = torch.sum(true_pos, dim=(1,2)) # [B, C]

                false_pos = torch.logical_and((preds_max == ones),(preds_max != y))
                false_pos = torch.sum(false_pos, dim=(1,2))

                false_neg = torch.logical_and((preds_max == zeros), (preds_max != y))
                false_neg = torch.sum(false_neg, dim=(1,2))
                          
                recall = self.compute_recall(true_pos, false_pos, false_neg)  # [B, C]
                total_recall_for_each_class += torch.sum(recall, dim=0) # [C]

                precision = self.compute_precision(true_pos, false_pos, false_neg)
                total_precision_for_each_class += torch.sum(precision, dim=0)  

        self.writer.add_scalar("Val/Average_loss",
                                total_loss/num_step,
                                self.training_step
                                )

        for idx, cls in enumerate(['other', 'road', 'vehicle']):
            self.writer.add_scalar(f"Val/recall_{cls}",
                                   total_recall_for_each_class[idx]/(num_step * self.loader.batch_size),
                                   self.training_step
                                  )
            self.writer.add_scalar(f"Val/precision_{cls}",
                                   total_precision_for_each_class[idx]/(num_step * self.loader.batch_size),
                                   self.training_step
                                  )

        network.train()


    def compute_recall(self, true_pos, false_pos, false_neg):
        return true_pos / (false_neg + true_pos + 1e-5)

    def compute_precision(self, true_pos, false_pos, false_neg):
        return true_pos / (true_pos + false_pos + 1e-5)

    def compute_iou(self, true_pos, false_pos, false_neg):
        return true_pos / (true_pos + false_pos + false_neg + 1e-5)



    def save_predicitons_as_img(self, network):
        network.eval()

        x, y = next(iter(self.loader))
        x = x.to(device=self.device)[:6]
        y = np.array(y, dtype=np.uint8)[:6]

        with torch.no_grad():
            preds = torch.argmax(network(x), dim=1, keepdim=False)
            preds_np = preds.cpu().numpy()
 
            for idx in range(preds_np.shape[0]):
                
                rgb = color_group[preds_np[idx]]
                self.writer.add_image(f'{idx}/predict', rgb/255., self.training_step, dataformats="HWC")

                rgb_mask_groundtruth = color_group[y[idx]]
                self.writer.add_image(f'{idx}/target', rgb_mask_groundtruth/255., self.training_step, dataformats="HWC")

        network.train()



