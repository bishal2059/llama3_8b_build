import torch
import torch.nn as nn


class TraderLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=500000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_sin_cos(self, seq_len, device):
        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum("n,d->nd", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        sin = emb.sin()[None, None, :, :]
        cos = emb.cos()[None, None, :, :]
        return sin, cos