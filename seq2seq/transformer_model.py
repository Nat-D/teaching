import torch
import torch.nn as nn
import random
from dataset import pad_idx, bos_idx, eos_idx
import math


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size, device, max_len = 500, dropout = 0.1,):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe = torch.zeros(max_len, 1, embedding_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class PositionalEncodingWithEmbedding(nn.Module):
    def __init__(self, embedding_size, device, max_len=500, dropout=0.1):
        super(PositionalEncodingWithEmbedding, self).__init__()

        self.embedding = nn.Embedding(max_len, embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.device  = device

    def forward(self, x):

        sequence_length, batch_size, embedding_size = x.shape
        positional_vectors = (torch.arange(0, sequence_length)
                              .unsqueeze(1)
                              .expand(sequence_length, batch_size)
                              .to(self.device)
                             ) # [seq_length, batch_size]

        x = x + self.embedding(positional_vectors)  # [seq_length, batch_size, embedding_size]
        return self.dropout(x)



class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 embedding_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward,
                 dropout,
                 device,
                 position_encoding_type='sine'):

        super(Seq2SeqTransformer, self).__init__()

        self.transformer = nn.Transformer(d_model=embedding_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(embedding_size, tgt_vocab_size)

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_size)

        if position_encoding_type == 'sine':
            self.src_positional_encoding = PositionalEncoding(embedding_size)
            self.tgt_positional_encoding = PositionalEncoding(embedding_size)

        elif position_encoding_type == 'learnable':
            self.src_positional_encoding = PositionalEncodingWithEmbedding(embedding_size, device)
            self.tgt_positional_encoding = PositionalEncodingWithEmbedding(embedding_size, device)


        self.device = device

    def forward(self, source, target, teacher_force_ratio=1.0):
        

        if teacher_force_ratio == 1.0:
            # x shape: (seq_length, batch)

            src_padding_mask = (source.transpose(0, 1) == pad_idx).to(self.device) 
            # src_padding_mask shape: (batch, seq_length)

            tgt_mask = self.transformer.generate_square_subsequent_mask(
                            target.shape[0]
                            ).to(self.device)
            # tgt_mask shape: (batch*num_heads, target_seq_length, target_seq_length)

            
            src_emb = self.src_positional_encoding(self.src_embedding(source))
            tgt_emb = self.tgt_positional_encoding(self.tgt_embedding(target))
            # embedding shape: (seq_length, batch, embedding_size)

            outs    = self.transformer(src=src_emb, 
                                       tgt=tgt_emb,
                                       tgt_mask=tgt_mask,
                                       src_key_padding_mask=src_padding_mask)

            return self.generator(outs)


        # greedy decode without target forcing
        elif teacher_force_ratio == 0.0:

            src_padding_mask = (source.transpose(0, 1) == pad_idx).to(self.device) 

            src_emb = self.src_positional_encoding(self.src_embedding(source))
            
            x = target[0].unsqueeze(0) # [1, batch_size]

            for i in range(target.shape[0]):

                tgt_emb = self.tgt_positional_encoding(self.tgt_embedding(x))

                tgt_mask = self.transformer.generate_square_subsequent_mask(
                            x.shape[0]
                            ).to(self.device)

                h = self.transformer(src=src_emb,
                                     tgt=tgt_emb,
                                     tgt_mask=tgt_mask,
                                     src_key_padding_mask=src_padding_mask,
                                     )  
                                    # [t_seq_length, batch, emb_size]

                gen_out = self.generator(h) # [t_seq_length, batch, vocab_size]

                # argmax along last feature / take the last token in seq
                best_guess = gen_out.argmax(dim=2)[-1, :].unsqueeze(0) # [1, batch_size]
                x = torch.cat([x, best_guess], dim=0)

            return gen_out

