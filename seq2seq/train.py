import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_iter, collate_fn, pad_idx
from torch.utils.data import DataLoader

from rnn_model import RnnEncoder, RnnDecoder, Seq2Seq
from logger import Logger

def main(num_epoch=10000,
         learning_rate=0.0001):
    
    # 1. get dataloader
    train_dataloader = DataLoader(train_iter, 
                        batch_size=6, 
                        collate_fn=collate_fn,
                        num_workers=2,
                        drop_last=True,
                        shuffle=True)

    # 2. model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_net = RnnEncoder().to(device)
    decoder_net = RnnDecoder().to(device)
    model = Seq2Seq(encoder_net, 
                          decoder_net).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. logger
    logger = Logger()

    # 4. training loop
    print('start training')
    for epoch in range(num_epoch):
        for german_batch, eng_batch in train_dataloader:
            german_batch = german_batch.to(device)
            eng_batch = eng_batch.to(device) # [seq_length, batch]

            # 4.1 prediction
            eng_pred = model(german_batch)

            # 4.2 compute loss
            loss = loss_fn(eng_pred, eng_batch)

            # 4.3 compute gradient 
            optimizer.zero_grad()
            loss.backward()

            # 4.4 update weights
            optimizer.step()

            logger.log_step(loss.item())
        
        logger.log_epoch(model) 
        

if __name__ == "__main__":
    main()