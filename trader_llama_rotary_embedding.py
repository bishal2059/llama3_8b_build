import torch
import torch.nn as nn

class TraderLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=131072, base=500000.0):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).float() / self.dim))
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build the cache (cos/sin tables)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=self.inv_freq.device, 
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids):
        seq_len = x.shape[-2] 
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[position_ids].unsqueeze(1),
            self.sin_cached[position_ids].unsqueeze(1),
        )