from torch.utils.tensorboard import SummaryWriter
from dataset import train_iter,   \
                    collate_fn,   \
                    pad_idx,      \
                    len_de_vocab, \
                    len_en_vocab

from torch.utils.data import DataLoader
import time
import numpy as np
from dataset import en_vocab_transform
from torchtext.datasets import Multi30k
import torch.nn as nn
import time
import torch

class Logger():
    def __init__(self, device, log_dir):
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.training_loss = 0
        self.training_step = 0
        self.num_steps_per_epoch = 0
        self.start_epoch_time = time.time()

        val_iter = Multi30k(split='valid')        
        self.val_dataloader = DataLoader(val_iter, 
                            batch_size=32, 
                            collate_fn=collate_fn,
                            num_workers=2,
                            drop_last=False,
                            shuffle=False)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
        self.itos = np.array(en_vocab_transform.get_itos())

    def log_step(self, loss):
        self.training_loss += loss
        self.training_step    += 1
        self.num_steps_per_epoch += 1


    def log_epoch(self, model):
        time_per_epoch = time.time() - self.start_epoch_time 
        self.start_epoch_time = time.time()

        self.writer.add_scalar('Training/time_per_epoch',
                                time_per_epoch,
                                self.training_step)

        self.writer.add_scalar('Training/Average_trainig_loss',
                                self.training_loss / self.num_steps_per_epoch,
                                self.training_step
                                )
        self.training_loss = 0
        self.num_steps_per_epoch = 0

        # validation
        self.validation(model)

    def validation(self, model):
        model.eval()
        total_loss = 0
        num_step = 0

        with torch.no_grad():
            for x,y in self.val_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds = model(source=x, 
                              target=y[:-1,:],
                              teacher_force_ratio=0.0)

                preds = preds.reshape(-1, preds.shape[2])
                y = y[1:].reshape(-1)

                loss  = self.loss_fn(preds, y)
                total_loss += loss
                num_step += 1 

            self.writer.add_scalar("Val/Average_loss",
                                   total_loss/num_step,
                                   self.training_step)

            x,y = next(iter(self.val_dataloader))


            x = x.to(self.device)
            y = y.to(self.device)
            preds = model(source=x, 
                          target=y[:-1,:],
                          teacher_force_ratio=0.0)
            best_guess = preds.argmax(2) #[seq_length, batch]

            best_guess = best_guess.cpu().numpy().astype(np.uint32)
        
            pred_eng_strings = np.transpose(self.itos[best_guess]).reshape(-1)
            pred_eng_strings = " ".join(pred_eng_strings)

            self.writer.add_text("prediction",
                                   pred_eng_strings,
                                   self.training_step)

        model.train()


