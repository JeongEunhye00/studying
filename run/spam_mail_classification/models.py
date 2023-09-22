import torch.nn as nn


class RNN_model(nn.Module):
    def __init__(self, batch_size, embedding_size, hidden_size, output_size, vocab_size, n_layers):
        super(RNN_model, self).__init__()

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          bias=False)
        self.fc = nn.Linear(hidden_size, output_size, bias = False) # W_y*h_t
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.rnn(x)
        output = self.softmax(self.fc(output[:, -1, :]))

        return output


class LSTM_model(nn.Module):
    def __init__(self, batch_size, embedding_size, hidden_size, output_size, vocab_size, n_layers):
        super(LSTM_model, self).__init__()

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.emb(x)
        output, (h_n, c_n) = self.lstm(x)
        output = self.softmax(self.fc(output[:, -1, :]))

        return output
