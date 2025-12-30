"""
TimeHandlerParallel Module

This module contains the TimeHandlerParallel class which handles parallel processing
of multiple time series bands with separate time encoders for each band.
"""

import torch
import torch.nn as nn
from ..encoders.positional_encoder import PositionalEncoder


class TimeHandlerParallel(nn.Module):
    """
    Parallel time handler for processing multiple time series bands.
    
    This class processes each band separately with its own time encoder,
    providing enhanced flexibility for multi-band time series data.
    
    Args:
        num_bands (int): Number of time series bands to process. Default: 2
        embedding_size (int): Size of the output embeddings. Default: 64
        Tmax (float): Maximum time value for normalization. Default: 3000.0
        **kwargs: Additional keyword arguments passed to encoders
    """
    
    def __init__(self, num_bands=2, embedding_size=64, Tmax=3000.0, **kwargs):
        super(TimeHandlerParallel, self).__init__()
        # general params
        self.num_bands = num_bands
        self.embedding_size = embedding_size
        self.T_max = Tmax
        self.time_encoders = nn.ModuleList([
            PositionalEncoder(embedding_size=embedding_size, **kwargs) 
            for _ in range(num_bands)
        ])

    def forward(self, x, t, mask=None, band_info=None, **kwargs):
        """
        Forward pass for parallel time handling.
        
        Args:
            x: Input magnitude tensor of shape [batch, seq]
            t: Time tensor of shape [batch, seq]
            mask: Optional mask tensor. If None, all positions are considered valid
            band_info: Band information tensor indicating which band each observation belongs to
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (x_mod, m_mod, t_mod) where:
                - x_mod: List of encoded tensors for each band
                - m_mod: List of mask tensors for each band
                - t_mod: List of time tensors for each band
        """
        # Verify mask is provided
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
            
        # Process each band
        x_mod, m_mod, t_mod = [], [], []
        batch, seq_len = x.shape 
        
        for i in range(1, self.num_bands + 1):
            x_mod_i = torch.zeros(batch, seq_len, self.embedding_size, device=x.device)
            m_mod_i = torch.zeros_like(mask)
            t_mod_i = torch.zeros_like(t)
            
            # get only the band i
            band_mask_2d = (band_info == i)  # Keep 2D mask for m_mod and t_mod
            band_mask_3d = band_mask_2d.unsqueeze(-1).expand(-1, -1, self.embedding_size)  # 3D mask for x_mod
            
            x_i = torch.where(band_mask_2d, x, torch.zeros_like(x))
            t_i = torch.where(band_mask_2d, t, torch.zeros_like(t))
            b_i = torch.where(band_mask_2d, band_info, torch.zeros_like(mask))
            
            mod = self.time_encoders[i-1](x_i, t_i, b_i)  # Note: i-1 since ModuleList is 0-indexed
            
            # Use appropriate masks for each tensor
            x_mod_ii = torch.where(band_mask_3d, mod, x_mod_i)
            x_mod.append(x_mod_ii)
            m_mod.append(torch.where(band_mask_2d, mask, m_mod_i).unsqueeze(-1))
            t_mod.append(torch.where(band_mask_2d, t, t_mod_i).unsqueeze(-1))

        return x_mod, m_mod, t_mod
