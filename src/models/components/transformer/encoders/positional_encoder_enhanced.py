"""
Most Advanced Positional Encoder Module

This module contains the PositionalEncoderEnhanced class which implements the most
comprehensive time series encoding with time differences, magnitude differences,
rate of change calculations, and multiple embedding strategies.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoderEnhanced(nn.Module):
    """
    Most advanced positional encoder with comprehensive time series features.
    
    This encoder combines multiple encoding strategies including:
    - Time and magnitude differences
    - Rate of change calculations
    - Sinusoidal time encoding
    - Convolutional and MLP-based embeddings
    - Learnable scaling factors
    
    Args:
        embedding_size (int): Output embedding dimension
        input_size (int): Input feature dimension. Default: 1
        seq_length (int): Maximum sequence length. Default: 2048
        num_bands (int): Number of photometric bands. Default: 2
        reduced_size_factor (int): Factor to reduce embedding dimensions for memory efficiency. Default: 2
        dropout (float): Dropout probability for regularization. Default: 0.1
        **kwargs: Additional keyword arguments
    """
    
    def __init__(self, embedding_size: int, input_size: int = 1, seq_length=2048, num_bands=2, 
                 reduced_size_factor=2, dropout=0.1, **kwargs):
        super(PositionalEncoderEnhanced, self).__init__()
        self.embedding_size = embedding_size
        self.seq_length = seq_length
        self.dropout = dropout
        
        # Use configurable reduced intermediate embedding size for efficiency
        # You can change reduced_size_factor to 4, 8, etc. for more memory savings
        reduced_size = embedding_size // reduced_size_factor
        self.reduced_size = reduced_size  # Store for potential debugging
        
        # All embedding dimensions now scale with reduced_size
        intermediate_size = reduced_size // 2  # For MLP hidden layers
        conv_out_size = embedding_size // 2    # Keep original conv size (not scaled by reduced_size_factor)
        
        self.band_embedding = nn.Embedding(num_embeddings=num_bands + 1, embedding_dim=reduced_size)

        # Time difference embedding - all sizes based on reduced_size
        self.time_diff_emb = nn.Sequential(
            nn.Linear(1, intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(intermediate_size, reduced_size)
        )
        
        # Magnitude difference embedding
        self.mag_diff_emb = nn.Sequential(
            nn.Linear(1, intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(intermediate_size, reduced_size)
        )
        
        # Rate of change embedding (magnitude_diff / time_diff)
        self.rate_emb = nn.Sequential(
            nn.Linear(1, intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(intermediate_size, reduced_size)
        )
        
        # Absolute time embedding (MLP-based)
        self.abs_time_emb = nn.Sequential(
            nn.Linear(1, intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(intermediate_size, reduced_size)
        )
        
        # Absolute magnitude embedding (MLP-based)
        self.abs_mag_emb = nn.Sequential(
            nn.Linear(1, intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(intermediate_size, reduced_size)
        )
        
        # Convolutional magnitude embedding
        self.magnitud_emb_big = nn.Conv1d(
            in_channels=1, 
            out_channels=conv_out_size,  # Now uses conv_out_size (= reduced_size)
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        #self.magnitud_emb_big_norm = nn.BatchNorm1d(conv_out_size)
        # Enhanced fusion with all embeddings (8 total: 7 with reduced_size + 1 conv with conv_out_size)
        total_input_size = reduced_size * 7 + conv_out_size  # 7 reduced_size embeddings + 1 conv embedding
        fusion_hidden_size = reduced_size * 2  # Scale fusion layer with reduced_size
        
        self.fusion_mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(self.dropout),
            #nn.RMSNorm(total_input_size),
            nn.Linear(total_input_size, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            #nn.RMSNorm(fusion_hidden_size),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, embedding_size),
            nn.RMSNorm(embedding_size)
        )
        
        # Small epsilon to avoid division by zero
        self.eps = 1e-8
        

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
    
    def compute_magnitude_differences(self, x):
        """
        Compute magnitude differences between consecutive steps.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            
        Returns:
            Magnitude differences of shape [batch, seq, 1] where first element is 0
        """
        batch_size, seq_len, _ = x.shape
        
        # Pad with zeros for the first magnitude step
        x_shifted = torch.cat([torch.zeros(batch_size, 1, 1, device=x.device), x[:, :-1]], dim=1)
        mag_diffs = x - x_shifted
        
        return mag_diffs
    
    def compute_rate(self, mag_diffs, time_diffs):
        """
        Compute rate of change: magnitude_diff / time_diff.
        
        Args:
            mag_diffs: Magnitude differences of shape [batch, seq, 1]
            time_diffs: Time differences of shape [batch, seq, 1]
            
        Returns:
            Rate of change of shape [batch, seq, 1]
        """
        # Avoid division by zero
        safe_time_diffs = time_diffs + self.eps 
        rates = mag_diffs / safe_time_diffs
        
        # Handle potential infinite values
        rates = torch.clamp(rates, min=-1e8, max=1e8)
        
        return rates

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
            Sinusoidal time embeddings of shape [batch, seq, reduced_size]
        """
        sin_emb = self.sinusoidal_time_encoding(t, self.reduced_size)
        return sin_emb

    def magnitud_embedding(self, x):
        """
        Encode magnitude values using 1D convolution.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            
        Returns:
            Magnitude embeddings of shape [batch, seq, reduced_size]
        """
        x = x.transpose(1, 2)
        x = self.magnitud_emb_big(x)
        #x = self.magnitud_emb_big_norm(x)
        x = x.transpose(1, 2)
        
        return x
    
    def forward(self, x, t, bands=None):
        """
        Forward pass with comprehensive time series encoding.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            t: Time tensor of shape [batch, seq, 1]
            bands: Band information (optional)
            
        Returns:
            Comprehensive embeddings of shape [batch, seq, embedding_size]
        """
        # Ensure proper dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
            
        # Compute differences and rates
        time_diffs = self.compute_time_differences(t)
        mag_diffs = self.compute_magnitude_differences(x)
        rates = self.compute_rate(mag_diffs, time_diffs)
        
        # Generate all embeddings
        t_emb = self.temporal_embedding(t)  # Sinusoidal time
        x_emb = self.magnitud_embedding(x)  # Convolutional magnitude

        # Difference embeddings with scaling
        time_diff_emb = self.time_diff_emb(time_diffs) 
        mag_diff_emb = self.mag_diff_emb(mag_diffs)
        
        # Rate embedding
        rate_emb = self.rate_emb(rates)

        # Absolute embeddings (alternative MLP-based representations)
        abs_time_emb = self.abs_time_emb(t) 
        abs_mag_emb = self.abs_mag_emb(x) 
        
        band_emb = self.band_embedding(bands.int()) 

        # Concatenate all 8 embeddings
        combined = torch.cat([
            t_emb,           # Absolute time (sinusoidal)
            x_emb,           # Absolute magnitude (conv)
            time_diff_emb,   # Time differences
            mag_diff_emb,    # Magnitude differences
            rate_emb,        # Rate of change (mag_diff / time_diff)
            abs_time_emb,    # Absolute time (MLP)
            abs_mag_emb,     # Absolute magnitude (MLP)
            band_emb,      # Band information (MLP)
        ], dim=2)
        
        # Final fusion through MLP
        out = self.fusion_mlp(combined)
        
        return out
