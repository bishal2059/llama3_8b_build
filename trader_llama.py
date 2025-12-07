from trader_llama_config import TraderLlamaConfig
from trader_llama_layer import TraderLlamaLayer
from trader_llama_rms_norm import TraderLlamaRMSNorm
import torch
import torch.nn as nn

class TraderLlama(nn.Module):
    def __init__(self,config: TraderLlamaConfig):
        super(TraderLlama, self).__init__()
        self.config = config
        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) 

        # Transformer layers
        self.layers = nn.ModuleList([TraderLlamaLayer(config) for _ in range(config.num_hidden_layers)])

        # Final layer norm
        self.norm = TraderLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_attention_heads)


    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, self.rotary_emb)

        x = self.norm(x)

        return x
        

