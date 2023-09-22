import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=-1)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        d_head = self.d_model // self.num_heads  # == d_k
        b_s, seq_len, _ = q.size()
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        # shape : (b_s, seq_len, d_model)

        Q = Q.view(b_s, -1, self.num_heads, d_head).permute(0, 2, 1, 3)
        K = K.view(b_s, -1, self.num_heads, d_head).permute(0, 2, 1, 3)
        V = V.view(b_s, -1, self.num_heads, d_head).permute(0, 2, 1, 3)
        # shape : (b_s, num_heads, seq_len, d_head)

        """
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k))V
        """
        score = Q.matmul(K.permute(0, 1, 3, 2)) / math.sqrt(d_head)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)
        weight = self.softmax(score)
        attention = weight.matmul(V)
        # shape : (b_s, num_heads, seq_len, d_head)

        x = attention.permute(0, 2, 1, 3).contiguous()
        # shape : (b_s, seq_len, num_heads, d_head)
        x = x.view(b_s, -1, self.d_model)
        # shape : (b_s, seq_len, d_model)
        x = self.W_o(x)
        # shape : (b_s, seq_len, d_model)

        return x, weight


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        x = (x-mean) / (std+self.eps)
        x = self.gamma * x + self.beta

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, prob):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x):
        x = self.W2(self.dropout(self.relu(self.W1(x))))

        return x
