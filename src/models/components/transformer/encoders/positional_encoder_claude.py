"""
Enhanced Positional Encoder with Time Differences Module

This module contains the PositionalEncoderClaude class which implements an enhanced
time and magnitude encoding with time differences and sinusoidal encoding.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoderClaude(nn.Module):
    """
    Enhanced Positional Encoder with time differences and sinusoidal encoding.
    
    This encoder combines convolutional embeddings with time difference information
    and sinusoidal positional encoding for improved temporal representation.
    
    Args:
        embedding_size (int): Output embedding dimension
        input_size (int): Input feature dimension. Default: 1
        seq_length (int): Maximum sequence length. Default: 2048
        **kwargs: Additional keyword arguments
    """
    
    def __init__(self, embedding_size: int, input_size: int = 1, seq_length=2048, **kwargs):
        super(PositionalEncoderClaude, self).__init__()
        self.embedding_size = embedding_size
        self.seq_length = seq_length
        
        # Time difference embedding
        self.time_diff_emb = nn.Sequential(
            nn.Linear(1, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, embedding_size)
        )
        
        # Absolute time embedding
        self.abs_time_emb = nn.Sequential(
            nn.Linear(1, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, embedding_size)
        )
        
        # Convolutional embeddings
        self.time_emb_big = nn.Conv1d(
            in_channels=1, 
            out_channels=embedding_size, 
            kernel_size=9, 
            stride=1, 
            padding=4
        )
        self.magnitud_emb_big = nn.Conv1d(
            in_channels=1, 
            out_channels=embedding_size, 
            kernel_size=9, 
            stride=1, 
            padding=4
        )
        
        # Enhanced fusion with time differences (4 embeddings total)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_size * 4, embedding_size * 2),
            nn.ReLU(),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.LayerNorm(embedding_size)
        )
        
        # Learnable scaling factors
        self.time_diff_scale = nn.Parameter(torch.ones(1))
        self.abs_time_scale = nn.Parameter(torch.ones(1))
        
    def compute_time_differences(self, t):
        """
        Compute time differences between consecutive steps.
        
        Args:
            t: Time tensor of shape [batch, seq, 1]
            
        Returns:
            Time differences of shape [batch, seq, 1] where first element is 0
        """
        batch_size, seq_len, _ = t.shape
        
        # Pad with zeros for the first time step
        t_shifted = torch.cat([torch.zeros(batch_size, 1, 1, device=t.device), t[:, :-1]], dim=1)
        time_diffs = t - t_shifted
        
        # Handle potential negative differences (safety check)
        time_diffs = torch.abs(time_diffs)
        
        return time_diffs
    
    def sinusoidal_time_encoding(self, t, d_model):
        """
        Apply sinusoidal encoding to time values for better representation.
        
        Args:
            t: Time tensor of shape [batch, seq, 1]
            d_model: Encoding dimension
            
        Returns:
            Sinusoidal encodings of shape [batch, seq, d_model]
        """
        batch_size, seq_len, _ = t.shape
        
        # Create position encodings
        div_term = torch.exp(torch.arange(0, d_model, 2, device=t.device) * 
                           -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(batch_size, seq_len, d_model, device=t.device)
        
        # Apply sin to even indices
        pe[:, :, 0::2] = torch.sin(t * div_term[None, None, :pe[:, :, 0::2].shape[-1]])
        
        # Apply cos to odd indices
        if d_model % 2 == 1:
            pe[:, :, 1::2] = torch.cos(t * div_term[None, None, :pe[:, :, 1::2].shape[-1]])
        else:
            pe[:, :, 1::2] = torch.cos(t * div_term[None, None, :])
            
        return pe
    
    def temporal_embedding(self, t):
        """
        Enhanced temporal embedding with sinusoidal encoding.
        
        Args:
            t: Time tensor of shape [batch, seq, 1]
            
        Returns:
            Enhanced time embeddings combining conv and sinusoidal features
        """
        # Original conv-based embedding
        t_transpose = t.transpose(1, 2)
        conv_emb = self.time_emb_big(t_transpose).transpose(1, 2)
        
        # Add sinusoidal encoding
        sin_emb = self.sinusoidal_time_encoding(t, self.embedding_size)
        
        return conv_emb + 0.1 * sin_emb  # Weighted combination
        
    def magnitud_embedding(self, x):
        """
        Encode magnitude values using 1D convolution.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            
        Returns:
            Magnitude embeddings of shape [batch, seq, embedding_size]
        """
        x = x.transpose(1, 2)
        x = self.magnitud_emb_big(x)
        x = x.transpose(1, 2)
        return x
    
    def forward(self, x, t, bands=None):
        """
        Forward pass with enhanced time and magnitude encoding.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            t: Time tensor of shape [batch, seq, 1]
            bands: Band information (optional)
            
        Returns:
            Enhanced embeddings of shape [batch, seq, embedding_size]
        """
        # Ensure proper dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
            
        # Compute time differences
        time_diffs = self.compute_time_differences(t)
        
        # Generate embeddings
        t_emb = self.temporal_embedding(t)  # Enhanced time with sinusoidal
        x_emb = self.magnitud_embedding(x)  # Magnitude
        
        # Time difference embedding
        diff_emb = self.time_diff_emb(time_diffs) * self.time_diff_scale
        
        # Absolute time embedding (alternative representation)
        abs_time_emb = self.abs_time_emb(t) * self.abs_time_scale
        
        # Concatenate all embeddings
        combined = torch.cat([t_emb, x_emb, diff_emb, abs_time_emb], dim=2)
        
        # Final fusion
        out = self.fusion_mlp(combined)
        
        return out
