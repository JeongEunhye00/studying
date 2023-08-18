# 구현

import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        h = self.tanh(self.W_x(x) + self.W_h(h))
        return h
