from trader_llama_config import TraderLlamaConfig
from trader_llama_layer import TraderLlamaLayer
from trader_llama_rms_norm import TraderLlamaRMSNorm
from trader_llama_rotary_embedding import TraderLlamaRotaryEmbedding
import torch
import torch.nn as nn

class TraderLlama(nn.Module):
    def __init__(self, config: TraderLlamaConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [TraderLlamaLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final RMSNorm
        self.norm = TraderLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Rotary embedding (shared across layers)
        self.rotary_emb = TraderLlamaRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False,
    ):

        B, T = input_ids.shape

        # ---- Embedding ----
        x = self.embed_tokens(input_ids)  # (B, T, hidden)

        # ---- Initialize cache if needed ----
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_past_key_values = [] if use_cache else None

        # ---- Transformer layers ----
        for layer, layer_past in zip(self.layers, past_key_values):
            x, new_layer_past = layer(
                x,
                rotary_emb=self.rotary_emb,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                new_past_key_values.append(new_layer_past)

        # ---- Final norm ----
        x = self.norm(x)

        return x, new_past_key_values

        

