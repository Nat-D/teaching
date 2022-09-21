class PositionalEncoding(nn.Module):
    def __init__(self):
        pass

    def forward(self, token_embedding):
        pass

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        pass
    def forward(self, tokens):
        pass


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward,
                 dropout):

        super(Seq2SeqTransformer, self).__init__()

        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)


    def forward(self, src, trg):

        tgt_mask, src_padding_mask = self.create_mask(src, trg) 

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs    = self.transformer(src=src_emb, 
                                   tgt=tgt_emb,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_padding_mask)
        return self.generator(outs)
  
    def encode(self):
        pass

    def decode(self):
        pass

