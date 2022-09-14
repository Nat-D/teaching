import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 p):

        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, 
                                      embedding_size)

        self.rnn = nn.LSTM(embedding_size,
                           hidden_size,
                           num_layers = 1,
                           bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size*2, 
                                   hidden_size)
        self.fc_cell   = nn.Linear(hidden_size*2,
                                   hidden_size)
        self.dropout   = nn.Dropout(p)

    def forward(self, x):

        # x: (seq_length, batch)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, batch, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # output shape: (seq_length, batch, hidden_size)
        # num_layers = 1
        # [bidirect] hidden shape: (2*num_layers, batch, hidden_size)
        # [bidirect] cell :        (2*num_layers, batch, hidden_size)

        # flatten hidden : (batch, 2*hidden_size) 
        hidden = self.fc_hidden(torch.cat(
                                (hidden[0:1], hidden[1:2])
                                ,dim=2))
        cell   = self.fc_cell(torch.cat(
                                (cell[0:1], cell[1:2])
                                ,dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 output_size,
                 p):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size,
                           hidden_size,
                           num_layers=1,
                           bidirectional=False)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu    = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):

        x = x.unsqueeze(0)
        # x: (1, batch_size)

        embedding = self.dropout(self.embedding(x))
        # embedding shape : (1, batch, embedding_size)

        # encoder_state shape: (seq_length, batch, hidden_size)
        # [bidirect] hidden shape: (2*num_layers, batch, hidden_size)
        sequence_length = encoder_states.shape[0] 

        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped : (sequence_length, batch, 2*hidden_size) 


        # note: attention energy comes from encoder hidden states
        #       this is a kind of self-attention.
        energy = self.relu(self.energy(
                    torch.cat((h_reshaped, encoder_states),dim=2) 
                    ))
        # energy shape: (seq_length, batch, 1)

        attention = self.softmax(energy)

        # attention: (seq_length, batch, 1) # sbk
        # encoder_states: (seq_length, batch, hidden_size*2) # sbl
        # we want context vector: (1, batch, hidden_size*2) # kbl
        context_vector = torch.einsum("sbk,sbl->kbl", attention, encoder_states)


        # sum of (self-attention apply on encoder_states) 
        # becomes a context of the whole sentence.
        # Then, context is concatenated with current input
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, batch, hidden*2 + embedding_size)


        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outpus shape: (1, batch_size, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (output_size, hidden_size)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, target_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size

        # get device from encoder
        self.device = next(self.encoder.parameters()).device


    def forward(self, source, target, teacher_force_ratio):
        
        batch_size = source.shape[1]
        target_len = target.shape[0]
        

        outputs = torch.zeros(target_len, batch_size, self.target_vocab_size).to(self.device)

        encoder_states, hidden, cell = self.encoder(source)
        x = target[0]

        for t in range(1, target_len):

            output, hidden, cell = self.decoder(x,
                                                encoder_states,
                                                hidden,
                                                cell)
            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess


        return outputs




















































        