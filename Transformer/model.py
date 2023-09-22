import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, n_layers, s_voca_len, t_voca_len, max_len, prob, s_pad_idx, t_pad_idx, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, d_ff, num_heads, n_layers, s_voca_len, max_len, prob, device)
        self.decoder = Decoder(d_model, d_ff, num_heads, n_layers, t_voca_len, max_len, prob, device)
        self.s_pad_idx = s_pad_idx
        self.t_pad_idx = t_pad_idx
        self.linear = nn.Linear(d_model, t_voca_len)
        self.softmax = nn.Softmax(dim=-1)

    def make_s_mask(self, src):
        s_mask = (src != self.s_pad_idx)  # <PAD>의 경우 0으로 처리해서 반영하지 않기 위함
        # shape : (b_s, seq_len)
        s_mask = s_mask.unsqueeze(1).unsqueeze(2)
        # shape : (b_s, 1, 1, seq_len)

        return s_mask

    def make_t_mask(self, trg):
        t_pad_mask = (trg != self.t_pad_idx)
        # shape : (b_s, seq_len)
        t_pad_mask = t_pad_mask.unsqueeze(1).unsqueeze(2)
        # shape : (b_s, 1, 1, seq_len)
        t_len = trg.size(1)
        t_sub_mask = torch.tril(torch.ones((t_len, t_len))).bool()  # 하부 삼각행렬
        t_mask = t_pad_mask & t_sub_mask  # <PAD>의 경우 행렬 전체를 False 로 처리

        return t_mask

    def forward(self, src, trg):
        s_mask = self.make_s_mask(src)
        t_mask = self.make_t_mask(trg)
        enc_output = self.encoder(src, s_mask)
        dec_output = self.decoder(enc_output, trg, s_mask, t_mask)
        output = self.softmax(self.linear(dec_output))

        return output
