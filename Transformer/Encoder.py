import torch.nn as nn
from Embedding import EmbeddingLayer
from layers import MultiHeadAttention, LayerNorm, FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, prob):
        super(EncoderLayer, self).__init__()
        self.m_h_att = MultiHeadAttention(num_heads, d_model)
        self.norm = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, prob)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x, mask):
        # multi-head attention
        x_ = x
        x, _ = self.m_h_att(q=x, k=x, v=x, mask=mask)
        # add & norm
        x = self.norm(x + x_)
        x = self.dropout(x)
        # feed forward
        x_ = x
        x = self.ffn(x)
        # add & norm
        x = self.norm(x + x_)
        x = self.dropout(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, n_layers, s_voca_len, max_len, prob, device):
        super(Encoder, self).__init__()
        self.emb_layer = EmbeddingLayer(d_model, s_voca_len, max_len, prob, device)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads, prob)\
                                         for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb_layer(x)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, s_mask)

        return x
