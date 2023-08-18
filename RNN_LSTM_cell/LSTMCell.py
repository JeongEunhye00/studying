# 구현

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Linear(self.input_size, 4 * self.hidden_size, bias=True)
        self.W_hh = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        gates = self.W_ih(x) + self.W_hh(h)
        f, i, g, o = gates.chunk(4, dim=1)
        f_t = self.sigmoid(f)
        i_t = self.sigmoid(i)
        g_t = self.tanh(g)
        o_t = self.sigmoid(o)

        c = torch.mul(f_t, c) + torch.mul(i_t, g_t)
        h = torch.mul(o_t, self.tanh(c))

        return h, c
