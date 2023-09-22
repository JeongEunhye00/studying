import torch.nn as nn
from Embedding import EmbeddingLayer
from layers import MultiHeadAttention, LayerNorm, FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, prob):
        super(DecoderLayer, self).__init__()
        self.m_h_att = MultiHeadAttention(num_heads, d_model)
        self.norm = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, prob)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, src, trg, s_mask, t_mask):
        # masked multi-head attention
        x_ = trg
        x, _ = self.m_h_att(q=trg, k=trg, v=trg, mask=t_mask)
        # add & norm
        x = self.norm(x + x_)
        x = self.dropout(x)
        # enc-dec multi-head attention
        x_ = x
        x, _ = self.m_h_att(q=x, k=src, v=src, mask=s_mask)
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


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, n_layers, t_voca_len, max_len, prob, device):
        super(Decoder, self).__init__()
        self.emb_layer = EmbeddingLayer(d_model, t_voca_len, max_len, prob, device)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads, prob)\
                                         for _ in range(n_layers)])

    def forward(self, x_s, x_t, s_mask, t_mask):
        x_t = self.emb_layer(x_t)
        for dec_layer in self.dec_layers:
            x_t = dec_layer(x_s, x_t, s_mask, t_mask)

        return x_t
