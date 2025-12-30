import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

class MultiheadAttention(nn.Module):
    """Optimized multi-head attention implementation."""
    
    def __init__(self, num_heads, embedding_size, dropout=0.1, **kwargs):
        super().__init__()
        
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.d_head = embedding_size // num_heads
        self.n_heads = num_heads
        self.dropout_p = dropout
        
        self.q_proj = nn.Linear(embedding_size, embedding_size, bias=True)
        self.k_proj = nn.Linear(embedding_size, embedding_size, bias=True)
        self.v_proj = nn.Linear(embedding_size, embedding_size, bias=True)      
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=True)
        
        # Store attention weights for visualization
        self.attn_weights = None
        self.save_attention = False

    def forward(self, x, mask=None, t=None, p=None):
        """Forward pass with optimized tensor operations."""
        batch_size, length, d_model = x.size()
        
        # Handle missing mask
        if mask is None:
            mask = torch.ones(batch_size, length, length, dtype=torch.bool, device=x.device)
            
        # Project inputs
        q_projected = self.q_proj(x)
        k_projected = self.k_proj(x)
        v_projected = self.v_proj(x)

        # Reshape for multi-head attention
        q = q_projected.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k_projected.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v_projected.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        mask = mask.bool().unsqueeze(1).permute(0, 1, 3, 2)
        
        # Calculate attention scores for visualization
        if self.save_attention:
            scale = 1.0 / (self.d_head ** 0.5)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(attn_scores.dtype).min)
            self.attn_weights = torch.softmax(attn_scores, dim=-1).detach()
        
        # Use optimized scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        
        out = out.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        out = self.out_proj(out)
        return out
        
    def get_attention_weights(self):
        """Return the attention weights from the last forward pass."""
        return self.attn_weights
    
class MultiheadLatentAttention(nn.Module):
    """Optimized multi-head attention implementation with latent space projections."""
    
    def __init__(self, num_heads, embedding_size, dropout=0.1, **kwargs):
        super().__init__()
        
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.d_head = embedding_size // num_heads
        self.n_heads = num_heads
        self.dropout_p = dropout
        d_model = embedding_size
        bias = True
        
        # Optimized: use half of model dimension for compressed representation
        d_c = d_model // 2
        d_c1 = d_model // 2
        
        self.DKV_proj = nn.Linear(d_model, d_c, bias=bias)
        self.DQ_proj = nn.Linear(d_model, d_c1, bias=bias)
        self.UQ_proj = nn.Linear(d_c1, d_model, bias=bias)
        self.UK_proj = nn.Linear(d_c, d_model, bias=bias)
        self.UV_proj = nn.Linear(d_c, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Store attention weights for visualization
        self.attn_weights = None
        self.save_attention = False

    def forward(self, x, mask=None, t=None, p=None):
        """Forward pass with optimized tensor operations."""
        batch_size, length, d_model = x.size()
        

        # Project to compressed dimension
        DQ = self.DQ_proj(x)
        DKV_key = self.DKV_proj(x)
        DKV_value = self.DKV_proj(x)

        # Project back to model dimension
        UQ = self.UQ_proj(DQ)
        UK = self.UK_proj(DKV_key)
        UV = self.UV_proj(DKV_value)

        # Reshape for multi-head attention
        UQ = UQ.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        UK = UK.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        UV = UV.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        mask = mask.bool().unsqueeze(1).permute(0, 1, 3, 2)
        
        # Calculate attention scores for visualization
        if self.save_attention:
            scale = 1.0 / (self.d_head ** 0.5)
            attn_scores = torch.matmul(UQ, UK.transpose(-2, -1)) * scale
            attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(attn_scores.dtype).min)
            self.attn_weights = torch.softmax(attn_scores, dim=-1).detach()

        # Use optimized scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
            query=UQ,
            key=UK,
            value=UV,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        out = self.out_proj(out)

        return out
        
    def get_attention_weights(self):
        """Return the attention weights from the last forward pass."""
        return self.attn_weights
    
class MultiheadPeriodicAttention(nn.Module):
    """Optimized multi-head attention with periodic bias."""
    
    def __init__(self, num_heads, embedding_size, dropout=0.1, **kwargs):
        super().__init__()
        
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.d_head = embedding_size // num_heads
        self.n_heads = num_heads
        self.kernel = kwargs.get("kernel", "periodic")
        self.kernel_size = kwargs.get("kernel_size", 1.0)
        self.dropout_p = dropout
        
        self.q_proj = nn.Linear(embedding_size, embedding_size, bias=True)
        self.k_proj = nn.Linear(embedding_size, embedding_size, bias=True)
        self.v_proj = nn.Linear(embedding_size, embedding_size, bias=True)      
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=True)
        
        # Store attention weights for visualization
        self.attn_weights = None
        self.save_attention = False
        self.periodic_bias = None

    def forward(self, x, mask=None, t=None, p=None,get_attn=False):
        """Forward pass with periodic attention.
        
        Args:
            query, key, value: Input tensors
            mask: Attention mask
            t: Timing information tensor
            p: Period tensor
        """
        # Error checking
        if t is None or p is None:
            raise ValueError("t and p must be provided for periodic attention")
        
        batch_size, length, d_model = x.size()
        
        # Handle missing mask
        if mask is None:
            mask = torch.ones(batch_size, length, length, dtype=torch.bool, device=x.device)
            
        # Project inputs
        q_projected = self.q_proj(x)
        k_projected = self.k_proj(x)
        v_projected = self.v_proj(x)

        # Reshape for multi-head attention
        q = q_projected.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k_projected.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v_projected.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Prepare mask
        mask = mask.bool().unsqueeze(1).permute(0, 1, 3, 2)
        mask = mask.expand(-1, self.n_heads, -1, -1)
        
        # Calculate attention scores with periodic adjustment
        scale = 1.0 / math.sqrt(self.d_head)
        q_scaled = q * scale
        attn = torch.matmul(q_scaled, k.transpose(-2, -1))
        
        if t is not None:
            periodic = self.periodic_attn(t.to(q.dtype), p.to(q.dtype)).sum(dim=-2, keepdim=True)
            self.periodic_bias = periodic.detach() if self.save_attention else None
            attn = attn - periodic
        
        # Apply mask and softmax
        attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)
        attn_weights = F.softmax(attn, dim=-1)
        

        attn_weights = F.dropout(attn_weights, p=self.dropout_p if self.training else 0.0, training=self.training)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        out = self.out_proj(out)
        
        return out, attn_weights
    
    @torch.compile
    def periodic_attn(self, t, p):
        """Compute periodic attention bias."""
        t_q = t[:, None, :, :]  # [batch, 1, seq_len, dim]
        t_k = t[:, None, :, :].transpose(-2, -1)  # [batch, 1, dim, seq_len]
        
        # Time differences
        L = t_q - t_k  # [batch, 1, seq_len, seq_len]
        
        # Zero out diagonal to prevent self-attention bias
        L = L.clone()  # Create a copy to avoid in-place modification issues
        L[:, :, 0, :] = 0.0
        
        # Convert periods to frequencies
        pi_div_p = (torch.pi / p)[:, :, None, None]  # [batch, dim, 1, 1]
        
        # Compute sin²(π·L/p)
        periodic = torch.sin(L * pi_div_p).pow(2)
        
        return periodic
        
    def get_attention_weights(self):
        """Return the attention weights from the last forward pass."""
        return self.attn_weights
        
    def get_periodic_bias(self):
        """Return the periodic bias from the last forward pass."""
        return self.periodic_bias
        
        # Compute sin²(π·L/p)
        periodic = torch.sin(L * pi_div_p).pow(2)
        
        return periodic

class MultiheadLatentPeriodicAttention(nn.Module):
    """Optimized multi-head attention with latent projections and periodic bias."""
    
    def __init__(self, num_heads, embedding_size, dropout=0.1, feat=False, **kwargs):
        super().__init__()
        
        # Handle the embedding size based on feat parameter
        embedding_size = kwargs.get("embedding_size", 512) if not feat else kwargs.get("embedding_size_feat", 512)
        
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.d_head = embedding_size // num_heads
        self.n_heads = num_heads
        self.dropout_p = dropout
        d_model = embedding_size
        bias = True
        
        # Compressed dimensions
        d_c = d_model // 2
        d_c1 = d_model // 2
        
        # Linear projections
        self.DKV_proj = nn.Linear(d_model, d_c, bias=bias)
        self.DQ_proj = nn.Linear(d_model, d_c1, bias=bias)
        self.UQ_proj = nn.Linear(d_c1, d_model, bias=bias)
        self.UK_proj = nn.Linear(d_c, d_model, bias=bias)
        self.UV_proj = nn.Linear(d_c, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, t=None, p=None):
        """Forward pass with latent periodic attention."""
        # Error checking
        if t is None or p is None:
            raise ValueError("t and p must be provided for periodic attention")
            
        batch_size, length, d_model = x.size()
        
        # Handle missing mask
        if mask is None:
            mask = torch.ones(batch_size, length, length, dtype=torch.bool, device=x.device)
        
        # Assume query=key=value for latent attention
        DQ = self.DQ_proj(x)
        DKV = self.DKV_proj(x)  # Same projection for both key and value

        # Project back to model dimension
        UQ = self.UQ_proj(DQ)
        UK = self.UK_proj(DKV)
        UV = self.UV_proj(DKV)

        # Reshape for multi-head attention
        UQ = UQ.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        UK = UK.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        UV = UV.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Prepare mask
        mask = mask.bool().unsqueeze(1).permute(0, 1, 3, 2)
        
        # Calculate attention scores with periodic adjustment
        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(UQ * scale, UK.transpose(-2, -1))
        if t is not None:
            periodic = self.periodic_attn(t.to(UQ.dtype), p.to(UQ.dtype)).sum(dim=-2, keepdim=True)
            attn = attn - periodic

        # Apply mask and softmax
        attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p if self.training else 0.0, training=self.training)
        
        # Apply attention to values
        out = torch.matmul(attn, UV)
        out = out.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        out = self.out_proj(out)
        
        return out
    
    @torch.compile
    def periodic_attn(self, t, p):
        """Compute periodic attention bias."""
        t_q = t[:, None, :, :]  # [batch, 1, seq_len, dim]
        t_k = t[:, None, :, :].transpose(-2, -1)  # [batch, 1, dim, seq_len]
        
        # Time differences
        L = t_q - t_k
        
        # Zero out diagonal to prevent self-attention bias
        L = L.clone()
        L[:, :, 0, :] = 0.0
        
        # Convert periods to frequencies
        pi_div_p = (torch.pi / p)[:, :, None, None]
        
        # Compute sin²(π·L/p)
        periodic = torch.sin(L * pi_div_p).pow(2)
        
        return periodic
    
class FlashAttentionMHA(nn.Module):
    def __init__(self, embedding_size, num_heads, max_seq_len=2048, dropout=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(embedding_size, 3 * embedding_size, bias=True)
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=True)
        self.qkv_proj.weight.data.normal_(0, 0.02)
        self.out_proj.weight.data.normal_(0, 0.02)
        
    def compute_attention_weights(self, q, k, mask):
        """
        Compute attention weights for visualization or analysis purposes.
        Since FlashAttention doesn't explicitly calculate the attention matrix,
        we need to compute it separately when needed.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            mask (torch.Tensor): Boolean mask of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Attention weights of shape [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = mask.shape
        
        # Adjust mask to match attention matrix shape (~mask means valid positions)
        attention_mask = (~mask).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        
        # Compute raw attention scores
        # Transpose q to [batch, num_heads, seq_len, head_dim]
        q_t = q.transpose(1, 2)
        # Transpose k to [batch, num_heads, head_dim, seq_len]
        k_t = k.transpose(1, 2).transpose(2, 3)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q_t, k_t) * scale  # [batch, num_heads, seq_len, seq_len]
        
        # Apply mask (set masked positions to large negative value)
        attn_scores = attn_scores.masked_fill(~attention_mask, torch.finfo(attn_scores.dtype).min)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights

    def forward(self, x,  mask, t=None, p=None,get_attn=False):
        batch_size, seq_len, _ = x.shape
        mask = mask.bool().squeeze(-1)
        mask = ~mask
        q, k, v = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).unbind(dim=2)
        seq_lens = (~mask).sum(dim=1).to(torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seq_lens, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = seq_lens.max().item()

        q_packed = q[~mask].reshape(-1, self.num_heads, self.head_dim)
        k_packed = k[~mask].reshape(-1, self.num_heads, self.head_dim)
        v_packed = v[~mask].reshape(-1, self.num_heads, self.head_dim)
        
        if get_attn:
            attn_weights = self.compute_attention_weights(q, k, mask)
        else:
            attn_weights = None

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
        
        return self.out_proj(out.to(x.dtype))
        
    def get_attention_weights(self, x, mask):
        """
        Get attention weights for the input tensors.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_size]
            mask (torch.Tensor): Mask tensor
            
        Returns:
            torch.Tensor: Attention weights
        """
        # Forward pass with get_attn=True to compute attention weights
        _, attn_weights = self.forward(x, mask, get_attn=True)
        return attn_weights


