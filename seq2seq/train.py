import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_iter,   \
                    collate_fn,   \
                    pad_idx,      \
                    len_de_vocab, \
                    len_en_vocab


from torch.utils.data import DataLoader
from rnn_model import Encoder, Decoder, Seq2Seq
from logger import Logger
import time


def main(num_epoch=10000,
         learning_rate=0.0001,
         batch_size=32):
    
    # 1. get dataloader
    train_dataloader = DataLoader(train_iter, 
                        batch_size=batch_size, 
                        collate_fn=collate_fn,
                        num_workers=2,
                        drop_last=True,
                        shuffle=True)

    # 2. model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_net = Encoder(input_size=len_de_vocab,
                             embedding_size=300,
                             hidden_size=1024,
                             num_layers=2,
                             p=0.5).to(device)
    decoder_net = Decoder(input_size=len_de_vocab,
                             embedding_size=300,
                             hidden_size=1024,
                             output_size=len_en_vocab,
                             num_layers=2,
                             p=0.5).to(device)

    model = Seq2Seq(encoder_net, 
                    decoder_net,
                    target_vocab_size=len_en_vocab).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. logger
    current_time = time.time()
    logger = Logger(device, log_dir=f'runs/{current_time}')

    # 4. training loop
    print('start training')
    for epoch in range(num_epoch):
        for german_batch, eng_batch in train_dataloader:
            german_batch = german_batch.to(device)
            eng_batch = eng_batch.to(device) # [seq_length, batch]

            # 4.1 prediction with teacher forcing
            eng_pred = model(source=german_batch, target=eng_batch)
            # eng_pred shape: (seq_length, batch, vocab_size)

            # reshape (should we also remove <bos> token?)
            eng_pred  = eng_pred.reshape(-1, eng_pred.shape[2])
            eng_batch = eng_batch.reshape(-1)

            # 4.2 compute loss
            loss = loss_fn(eng_pred, eng_batch)

            # 4.3 compute gradient 
            optimizer.zero_grad()
            loss.backward()

            # (optional) clip norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # 4.4 update weights
            optimizer.step()

            logger.log_step(loss.item())
        
        logger.log_epoch(model) 
        

if __name__ == "__main__":
    main()