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
    def __init__(self, embedding_size:int = 512, prenorm: bool = False, norm_type: str = "layernorm",drop_path: float = 0.1, **kwargs):
        super().__init__()
        self.norm1 = get_norm_layer(norm_type, embedding_size)
        self.norm2 = get_norm_layer(norm_type, embedding_size)
        self.attn = FlashAttentionMHA(embedding_size=embedding_size, **kwargs)
        self.prenorm = prenorm
        self.ff = FeedForward(embedding_size=embedding_size, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, mask: Optional[Tensor] = None, t=None, p=None,get_attn=False) -> Tensor:
        # initialize x from query input
        y = self.attn(self.norm1(x), mask, t, p,get_attn=get_attn)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.ff(self.norm2(x)))
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
       #x = self.last_norm(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)