from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from .MultiheadAttention import MultiheadAttention, MultiheadLatentAttention,MultiheadLatentPeriodicAttention, MultiheadPeriodicAttention,FlashAttentionMHA


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.1, expansion_coef: int = 2, **kwargs):
        super().__init__()
        embedding_size_sub = embedding_size * expansion_coef
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size_sub),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size_sub, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size:int = 512, prenorm: bool = False, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.attn = FlashAttentionMHA(embedding_size=embedding_size, **kwargs)
        self.ff = FeedForward(embedding_size=embedding_size, **kwargs)

    def forward(self, x, mask: Optional[Tensor] = None, t=None, p=None,get_attn=False) -> Tensor:
        x = self.attn(self.norm1(x), mask, t, p,get_attn=get_attn) + x
        x = self.ff(self.norm2(x)) + x
        return x

class Transformer(nn.Module):
    def __init__(self, num_encoders: int = 3, embedding_size: int = 512, **kwargs):
        super().__init__()
        self.stacked_encoders = nn.ModuleList([TransformerEncoder(num_encoders=num_encoders, embedding_size=embedding_size, **kwargs) for _ in range(num_encoders)])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, t=None, p=None,get_attn=False) -> Tensor:
        for encoder in self.stacked_encoders:
            x = encoder(x, mask, t, p, get_attn=get_attn)
        return x[:,0,:]
