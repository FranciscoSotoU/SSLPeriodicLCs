"""
TimeHandler Module

This module contains the TimeHandler class which handles time series encoding
with configurable encoder types and band processing.
"""

import torch
import torch.nn as nn
from ..encoders.time_film import TimeFilm
from ..encoders.positional_encoder import PositionalEncoder
from ..encoders.positional_encoder_enhanced import PositionalEncoderEnhanced
from ..encoders.positional_encoder_enhanced_experimental import PositionalEncoderEnhancedE


class TimeHandler(nn.Module):
    """
    Configurable time handler for processing time series data.
    
    This class supports multiple encoder types and processes bands sequentially,
    providing flexibility in time series encoding strategies.
    
    Args:
        num_bands (int): Number of time series bands to process. Default: 2
        embedding_size (int): Size of the output embeddings. Default: 64
        Tmax (float): Maximum time value for normalization. Default: 3000.0
        **kwargs: Additional keyword arguments including:
            - time_encoder (str): Type of encoder to use. Options: 'TimeFilm', 
              'PositionalEncoder', 'PositionalEncoderEnhanced'. Default: 'PositionalEncoderEnhanced'
            - input_size (int): Input feature dimension. Default: 1
            - seq_length (int): Maximum sequence length. Default: 200
    """
    
    def __init__(self, num_bands=2, embedding_size=64, Tmax=3000.0, **kwargs):
        super(TimeHandler, self).__init__()
        # general params
        time_encoder = kwargs.get("time_encoder", "PositionalEncoderEnhanced")

        self.num_bands = num_bands
        self.embedding_size = embedding_size
        self.T_max = Tmax
        
        if time_encoder == "TimeFilm":
            self.time_encoders = nn.ModuleList([
                TimeFilm(embedding_size=embedding_size, Tmax=Tmax, **kwargs) 
                for _ in range(num_bands)
            ])
        elif time_encoder == 'PositionalEncoder':
            self.time_encoders = nn.ModuleList([
                PositionalEncoder(
                    embedding_size=embedding_size, 
                    input_size=kwargs.get("input_size", 1), 
                    seq_length=kwargs.get("seq_length", 200)
                ) for _ in range(num_bands)
            ])
        elif time_encoder == 'PositionalEncoderEnhanced':
            self.time_encoders = nn.ModuleList([
                PositionalEncoderEnhanced(
                    embedding_size=embedding_size, 
                    input_size=kwargs.get("input_size", 1)
                ) for _ in range(num_bands)
            ])
        elif time_encoder == 'PositionalEncoderEnhancedE':
            self.time_encoders = nn.ModuleList([
                PositionalEncoderEnhancedE(
                    embedding_size=embedding_size, 
                    input_size=kwargs.get("input_size", 1)
                ) for _ in range(num_bands)
            ])
    def forward(self, x, t, mask=None, band_info=None, **kwargs):
        """
        Forward pass for time handling with band processing.
        
        Args:
            x: Input magnitude tensor of shape [batch, seq]
            t: Time tensor of shape [batch, seq]
            mask: Optional mask tensor. If None, all positions are considered valid
            band_info: Band information tensor indicating which band each observation belongs to
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (x_mod, mask, t) where:
                - x_mod: Encoded tensor of shape [batch, seq, embedding_size]
                - mask: Reshaped mask tensor of shape [batch, seq, 1]
                - t: Reshaped time tensor of shape [batch, seq, 1]
        """
        batch, seq_len = x.shape

        # collect per‐band outputs
        x_mod = torch.zeros(batch, seq_len, self.embedding_size, device=x.device)
        
        for i in range(1, self.num_bands + 1):
            # Create band mask expanded to match dimensions
            band_mask = (band_info == i).unsqueeze(-1).expand(-1, -1, self.embedding_size)
            
            x_i = torch.where(band_info == i, x, torch.zeros_like(x))
            t_i = torch.where(band_info == i, t, torch.zeros_like(t))
            b_i = torch.where(band_info == i, band_info, torch.zeros_like(mask))

            # Sort by band information for processing
            index = torch.argsort(b_i, dim=1, descending=True)
            x_i = torch.gather(x_i, 1, index)
            t_i = torch.gather(t_i, 1, index)
            b_i = torch.gather(b_i, 1, index)

            mod = self.time_encoders[i-1](x_i, t_i, b_i)  # Note: i-1 since ModuleList is 0-indexed
            
            # Inverse the index to get back to original order
            # Compute inverse permutation
            inverse_index = torch.zeros_like(index)
            inverse_index.scatter_(1, index, torch.arange(index.size(1), device=index.device).unsqueeze(0).expand_as(index))

            # Reorder mod back to original
            mod = torch.gather(mod, 1, inverse_index.unsqueeze(-1).expand(-1, -1, self.embedding_size))

            # Use the expanded mask for element-wise selection
            x_mod = torch.where(band_mask, mod, x_mod)
            
        mask = mask.view(batch, seq_len, 1)
        t = t.view(batch, seq_len, 1)

        return x_mod, mask, t
