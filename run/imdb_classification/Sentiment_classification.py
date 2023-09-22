import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import torch.nn as nn
from Transformer.Encoder import Encoder


class SentimentCLS(nn.Module):
    def __init__(self, s_pad_idx, vocab_len, d_model, num_heads, d_ff, n_layers, prob, max_len, n_class, device):
        super().__init__()
        self.s_pad_idx = s_pad_idx
        self.encoder = Encoder(d_model, d_ff, num_heads, n_layers, vocab_len, max_len, prob, device)
        self.fc_layer = nn.Sequential(nn.Linear(max_len*d_model, d_model),
                                      nn.BatchNorm1d(d_model),
                                      nn.ReLU(),
                                      nn.Dropout(prob),
                                      nn.Linear(d_model, d_model//2),
                                      nn.BatchNorm1d(d_model//2),
                                      nn.ReLU(),
                                      nn.Dropout(prob),
                                      nn.Linear(d_model//2, n_class),
                                      nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

    def make_s_mask(self, src):
        s_mask = (src != self.s_pad_idx)  # <PAD>의 경우 0으로 처리해서 반영하지 않기 위함
        # shape : (b_s, seq_len)
        s_mask = s_mask.unsqueeze(1).unsqueeze(2)
        # shape : (b_s, 1, 1, seq_len)

        return s_mask

    def forward(self, src):
        s_mask = self.make_s_mask(src)
        enc_out = self.encoder(src, s_mask=s_mask)
        flat_enc_out = enc_out.view(enc_out.size(0), -1).contiguous()  # (b_s, max_len * d_model)
        output = self.softmax(self.fc_layer(flat_enc_out))

        return output
