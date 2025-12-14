from trader_llama_config import TraderLlamaConfig
from trader_llama_rms_norm import TraderLlamaRMSNorm
import math
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class KVCache:
    k: torch.Tensor  
    v: torch.Tensor  

    @property
    def seq_len(self):
        return self.k.size(2)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
 
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class TraderLlamaLayer(nn.Module):
    def __init__(self, config: TraderLlamaConfig):
        super(TraderLlamaLayer, self).__init__()
        self.config = config
        self.self_attn = TraderLlamaAttention(config)
        self.mlp = TraderLlamaMLP(config)
        self.input_layernorm = TraderLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TraderLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, rotary_emb):
        # Attention block
        normed_x = self.input_layernorm(x)
        attn_output = self.self_attn(normed_x, rotary_emb)
        x = x + attn_output

        # MLP block
        normed_x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output

        return x
    
class TraderLlamaMLP(nn.Module):
    def __init__(self, config: TraderLlamaConfig):
        super(TraderLlamaMLP, self).__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        if config.hidden_activation == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_activation}")

    def forward(self, x):
        gate_output = self.gate_proj(x)
        up_proj_output = self.up_proj_proj(x)
        return self.down_proj(self.act_fn(gate_output) * up_proj_output)
    
class TraderLlamaAttention(nn.Module):
    def __init__(self, config: TraderLlamaConfig):
        super(TraderLlamaAttention, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (self.head_dim * self.num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scale = 1 / (self.head_dim ** 0.5)
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self,
        x,
        attn_mask=None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ):
        """
        x: (B, T, E)
        kv_cache: KVCache or None
        use_cache: whether to return updated cache
        """

        B, T, _ = x.shape
        device = x.device

        # ---- Projections ----
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---- Reshape ----
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ---- Rotary embeddings ----
        if self.rotary_dim > 0:
            past_len = kv_cache.seq_len if kv_cache is not None else 0
            sin, cos = self.rotary.get_sin_cos(past_len + T, device)

            sin = sin[:, :, past_len:, :]
            cos = cos[:, :, past_len:, :]

            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

            q_rot, k_rot = apply_rotary(q_rot, k_rot, sin, cos)

            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        # ---- KV Cache update ----
        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)

        new_cache = KVCache(k, v) if use_cache else None

        # ---- Expand KV to heads ----
        if self.num_kv_heads != self.num_heads:
            idx = torch.arange(self.num_heads, device=device) % self.num_kv_heads
            k = k[:, idx]
            v = v[:, idx]

        # ---- Attention ----
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, self.dropout, self.training)

        out = torch.matmul(attn_probs, v)

        # ---- Merge heads ----
        out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
        out = self.out_proj(out)

        return out, new_cache

    
