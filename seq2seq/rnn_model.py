import torch.nn as nn
import random
import torch


class RnnEncoder(nn.Module):
    def __init__(self, 
                 input_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 p):

        super(RnnEncoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn       = nn.LSTM(embedding_size,
                                 hidden_size,
                                 num_layers,
                                 dropout=p)
    def forward(self, x):
        # x shape: (seq_length, batch)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, batch, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # output shape: (seq_length, batch, hidden_size)
        # hidden shape: (num_layers, hidden_size)
        # cell :        (num_layers, hidden_size)

        return hidden, cell


class RnnDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 p):
        super(RnnDecoder, self).__init__()
        
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size,
                           hidden_size,
                           num_layers,
                           dropout=p)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):

        # x shape: (batch_size), we want (seq_length=1, batch_size)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding_shape: (1, batch_size, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, batch_size, hidden_size)

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        # predictions shape: (1, batch_size, output_size)
        # we want (batch_size, output_size)
        # note: output_size = number of words in vocab 
        return predictions, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder,
                 target_vocab_size):
    
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.target_vocab_size = target_vocab_size

        # get device from encoder
        self.device = next(self.encoder.parameters()).device

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        
        outputs = torch.zeros(target_len, batch_size, self.target_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(source)
        x = target[0]

 
        # start with (x, hidden, cell) 
        # where x=<BOS> and (hidden,cell) is the context from encoder.
        for t in range(1, target_len):

            output, hidden, cell = self.decoder(x, hidden, cell)
            # output shape: (batch_size, vocab_size)

            # store next output prediction
            outputs[t] = output

            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs














