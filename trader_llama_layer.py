from trader_llama_config import TraderLlamaConfig
from trader_llama_rms_norm import TraderLlamaRMSNorm
import torch
import torch.nn as nn


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
        attn_output = self.attention(normed_x, rotary_emb)
        x = x + attn_output

        # MLP block
        normed_x = self.post_attention_norm(x)
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