"""
Basic Positional Encoder Module

This module contains the PositionalEncoder class which implements convolutional-based
time and magnitude encoding for time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    """
    Basic Positional Encoder using convolutional layers.
    
    This encoder uses 1D convolutions to encode both time and magnitude values,
    then fuses them through an MLP to create position-aware embeddings.
    
    Args:
        embedding_size (int): Output embedding dimension
        input_size (int): Input feature dimension. Default: 1
        seq_length (int): Maximum sequence length. Default: 2048
        **kwargs: Additional keyword arguments
    """
    
    def __init__(self, embedding_size: int, input_size: int = 1, seq_length=2048, **kwargs):
        super(PositionalEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.seq_length = seq_length
        
        # Convolutional encoders for time and magnitude
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
        
        # Fusion MLP to combine time and magnitude embeddings
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def temporal_embedding(self, t):
        """
        Encode time values using 1D convolution.
        
        Args:
            t: Time tensor of shape [batch, seq, 1]
            
        Returns:
            Time embeddings of shape [batch, seq, embedding_size]
        """
        # Transpose for Conv1d: [batch, seq, 1] -> [batch, 1, seq]
        t = t.transpose(1, 2)
        t = self.time_emb_big(t)
        # Transpose back to [batch, seq, embedding_size]
        t = t.transpose(1, 2)
        return t

    def magnitud_embedding(self, x):
        """
        Encode magnitude values using 1D convolution.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            
        Returns:
            Magnitude embeddings of shape [batch, seq, embedding_size]
        """
        # Transpose for conv1d: [batch, seq, 1] -> [batch, 1, seq]
        x = x.transpose(1, 2)
        x = self.magnitud_emb_big(x)
        x = x.transpose(1, 2)
        return x

    def forward(self, x, t, bands):
        """
        Forward pass combining time and magnitude embeddings.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            t: Time tensor of shape [batch, seq, 1]
            bands: Band information (unused in this implementation)
            
        Returns:
            Combined embeddings of shape [batch, seq, embedding_size]
        """
        # Ensure proper dimensions
        t_emb = self.temporal_embedding(t.unsqueeze(-1))
        x_emb = self.magnitud_embedding(x.unsqueeze(-1))
        
        # Concatenate and fuse through MLP
        out = self.fusion_mlp(torch.cat([t_emb, x_emb], dim=2))
        return out
