import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    RoPE encodes positional information by rotating query and key embeddings
    based on their absolute positions in a way that naturally incorporates
    relative positional information into the attention mechanism.
    """
    
    def __init__(self, head_dim, max_seq_len=512, base=1000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute frequency matrix
        self._build_cache(max_seq_len)
        
    def _build_cache(self, max_seq_len):
        """Build cos/sin cache for RoPE."""
        # Create frequency for each dimension pair
        # Use head_dim instead of head_dim//2 for the denominator
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        
        # Create position indices
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # Compute frequencies for each position
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim // 2]
        
        # Compute cos and sin
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)
        
    def _apply_rope(self, x, cos, sin):
        """Apply rotary embedding using cos/sin decomposition."""
        # Split x into two halves
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,  # Real part
            x1 * sin + x2 * cos   # Imaginary part
        ], dim=-1)
        
        # Flatten back to original shape
        return rotated.flatten(-2)
        
    def apply_rope(self, x, start_pos=0):
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            start_pos: Starting position offset (useful for generation)
            
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        # Extend cache if needed
        if start_pos + seq_len > self.freqs_cos.size(0):
            self._build_cache(start_pos + seq_len)
        
        # Get relevant frequencies
        cos = self.freqs_cos[start_pos:start_pos + seq_len]  # [seq_len, head_dim // 2]
        sin = self.freqs_sin[start_pos:start_pos + seq_len]  # [seq_len, head_dim // 2]
        
        # Expand dimensions for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim // 2]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim // 2]
        
        # Apply rotation
        return self._apply_rope(x, cos, sin)



class FlashAttentionMHA(nn.Module):
    def __init__(self, embedding_size, num_heads, max_seq_len=2048, dropout=0.1, use_rope=False, use_proj_drop=False, **kwargs):
        super().__init__()
        self.embed_dim = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.dropout = dropout
        self.use_rope = use_rope
        
        # Ensure head_dim is even for RoPE
        if use_rope and self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        
        self.qkv_proj = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.qkv_proj.weight.data.normal_(0, 0.02)
        self.out_proj.weight.data.normal_(0, 0.02)
        
        # Initialize RoPE if enabled
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.use_proj_drop = use_proj_drop
        if use_proj_drop:
            
            self.proj_drop = nn.Dropout(dropout)

    def forward(self, x,  mask, t=None, p=None,get_attn=False):
        batch_size, seq_len, _ = x.shape
        mask = mask.bool().squeeze(-1)
        mask = ~mask
        q, k, v = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).unbind(dim=2)
        
        # Apply RoPE to query and key tensors
        if self.use_rope:
            q = self.rope.apply_rope(q)
            k = self.rope.apply_rope(k)
        
        seq_lens = (~mask).sum(dim=1).to(torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seq_lens, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = seq_lens.max().item()

        q_packed = q[~mask].reshape(-1, self.num_heads, self.head_dim)
        k_packed = k[~mask].reshape(-1, self.num_heads, self.head_dim)
        v_packed = v[~mask].reshape(-1, self.num_heads, self.head_dim)
        

        out_packed = flash_attn_varlen_func(
            q_packed.to(torch.bfloat16), 
            k_packed.to(torch.bfloat16), 
            v_packed.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,

        )

        out = torch.zeros(batch_size, seq_len, self.num_heads, self.head_dim, 
                         device=x.device, dtype=torch.bfloat16)
        out[~mask] = out_packed
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        if self.use_proj_drop:
            out = self.proj_drop(out)

        return self.out_proj(out.to(x.dtype))
