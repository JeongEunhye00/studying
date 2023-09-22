import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(max_len, device=device).unsqueeze(1)  # [[0,], [1,], ... , [max_len-1,]]
        _2i = torch.arange(0, d_model, 2, device=device)  # [0, 2, 4, ...] => 2i

        self.pe = torch.zeros(max_len, d_model, device=device)
        self.pe.requires_grad = False
        self.pe[:, 0::2] = torch.sin(pos / (10000.0 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000.0 ** (_2i / d_model)))

    def forward(self, x):
        """" x shape : (b_s, seq_len, d_model) """
        seq_len = x.size(1)

        return self.pe[:seq_len, :]


class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, vocab_len, max_len, prob, device):
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.word_emb = nn.Embedding(vocab_len, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x):
        word_emb = self.word_emb(x)
        pos_emb = self.pos_emb(x)

        return self.dropout(word_emb + pos_emb)
