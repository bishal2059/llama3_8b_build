from trader_llama_config import TraderLlamaConfig
import torch
import torch.nn as nn

class TraderLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super(TraderLlamaRMSNorm, self).__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
    
    def forward(self, x):
        return self.norm(x)