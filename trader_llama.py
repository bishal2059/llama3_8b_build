from model_config import TraderLlamaConfig
import torch
import torch.nn as nn


class TraderLlama:
    def __init__(self,config: TraderLlamaConfig):
        self.config = config
        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) 

        # Transformer layers
        self.layers = nn.ModuleList([TraderLllamaLayer(config) for _ in range(config.num_hidden_layers)])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output layer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
