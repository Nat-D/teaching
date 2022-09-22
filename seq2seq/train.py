import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_iter,   \
                    collate_fn,   \
                    pad_idx,      \
                    len_de_vocab, \
                    len_en_vocab


from torch.utils.data import DataLoader
from logger import Logger
import time
import rnn_model, rnn_attention_model
import transformer_model

# model factory
def create_model(model_type, device):
    model = None
    if model_type == 'rnn':  
        encoder_net = rnn_model.Encoder(input_size=len_de_vocab,
                                 embedding_size=512,
                                 hidden_size=1024,
                                 num_layers=2,
                                 p=0.5).to(device)
        decoder_net = rnn_model.Decoder(input_size=len_en_vocab,
                                 embedding_size=512,
                                 hidden_size=1024,
                                 output_size=len_en_vocab,
                                 num_layers=2,
                                 p=0.5).to(device)
        model = rnn_model.Seq2Seq(encoder_net, 
                                 decoder_net,
                                 target_vocab_size=len_en_vocab).to(device)
    
    elif model_type == "attention":
        
        encoder_net =  rnn_attention_model.Encoder(
                                input_size = len_de_vocab,
                                embedding_size = 512,
                                hidden_size = 1024,
                                p=0.5).to(device)
        decoder_net = rnn_attention_model.Decoder(
                                input_size = len_en_vocab,
                                embedding_size = 512,
                                hidden_size = 1024,
                                output_size = len_en_vocab,
                                p=0.5).to(device)
        model = rnn_attention_model.Seq2Seq(
                                encoder_net,
                                decoder_net,
                                target_vocab_size=len_en_vocab).to(device)

    elif model_type == "transformer":

        model = transformer_model.Seq2SeqTransformer( 
                                 num_encoder_layers = 3,
                                 num_decoder_layers = 3,
                                 embedding_size = 512,
                                 nhead = 8,
                                 src_vocab_size = len_de_vocab,
                                 tgt_vocab_size = len_en_vocab,
                                 dim_feedforward = 1024,
                                 dropout = 0.1,
                                 device = device).to(device)

    return model


def main(num_epoch=15,
         learning_rate=0.0001,
         batch_size=32,
         model_type="attention"):
    
    # 1. get dataloader
    train_dataloader = DataLoader(train_iter, 
                        batch_size=batch_size, 
                        collate_fn=collate_fn,
                        num_workers=2,
                        drop_last=True,
                        shuffle=True)

    # 2. model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type, device=device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. logger
    current_time = time.time()
    logger = Logger(device, 
                    log_dir=f'runs/{model_type}{current_time}')

    # 4. training loop
    print('start training')
    for epoch in range(num_epoch):
        for german_batch, eng_batch in train_dataloader:
            
            german_batch = german_batch.long().to(device)
            eng_batch = eng_batch.long().to(device) # [seq_length, batch]

            # 4.1 prediction with teacher forcing / remove <eos> target token
            eng_pred = model(source=german_batch, target=eng_batch[:-1,:])
            # eng_pred shape: (seq_length, batch, vocab_size)

            eng_pred  = eng_pred.reshape(-1, eng_pred.shape[2])
            
            # shift the target 
            eng_batch = eng_batch[1:].reshape(-1)

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

    main(num_epoch=20, model_type="transformer")
    main(num_epoch=20, model_type="attention")
    main(num_epoch=20, model_type="rnn")