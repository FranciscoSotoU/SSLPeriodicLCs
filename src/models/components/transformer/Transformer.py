from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from .MultiheadAttention import FlashAttentionMHA


def get_norm_layer(norm_type: str, embedding_size: int) -> nn.Module:
    """
    Factory function to create normalization layers based on type.
    
    Args:
        norm_type: Type of normalization ('layernorm' or 'rmsnorm')
        embedding_size: Size of the embedding dimension
        
    Returns:
        Normalization layer instance
    """
    if norm_type.lower() == "layernorm":
        return nn.LayerNorm(embedding_size)
    elif norm_type.lower() == "rmsnorm":
        return nn.RMSNorm(embedding_size)
    
class FeedForward(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.1, expansion_coef: int = 4,bias: bool = False, **kwargs):
        super().__init__()
        embedding_size_sub = embedding_size * expansion_coef
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size_sub, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size_sub, embedding_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size:int = 512, prenorm: bool = False, norm_type: str = "layernorm", **kwargs):
        super().__init__()
        self.norm1 = get_norm_layer(norm_type, embedding_size)
        self.norm2 = get_norm_layer(norm_type, embedding_size)

        self.attn = FlashAttentionMHA(embedding_size=embedding_size, **kwargs)
        self.prenorm = prenorm
        self.ff = FeedForward(embedding_size=embedding_size, **kwargs)

    def forward(self, x, mask: Optional[Tensor] = None, t=None, p=None,get_attn=False) -> Tensor:
        # initialize x from query input
        
        x = self.attn(self.norm1(x), mask, t, p,get_attn=get_attn) + x
        x = self.ff(self.norm2(x)) + x
        return x

class Transformer(nn.Module):
    def __init__(self, num_encoders: int = 3, embedding_size: int = 512, norm_type: str = "layernorm", **kwargs):
        super().__init__()
        self.stacked_encoders = nn.ModuleList([
            TransformerEncoder(
                num_encoders=num_encoders, 
                embedding_size=embedding_size, 
                norm_type=norm_type,
                **kwargs
            ) for _ in range(num_encoders)
        ])
        #self.last_norm = get_norm_layer(norm_type, embedding_size)
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, t=None, p=None,get_attn=False) -> Tensor:
        for encoder in self.stacked_encoders:
            x= encoder(x, mask, t, p, get_attn=get_attn)

        return x
