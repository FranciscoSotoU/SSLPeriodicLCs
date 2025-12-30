"""
TimeFilm Encoder Module

This module contains the TimeFilm class which implements Fourier-based time encoding
using learnable harmonic parameters for time series data.
"""

import torch
import torch.nn as nn


class TimeFilm(nn.Module):
    """
    Time-based Feature Learning with Fourier coefficients (TimeFilm).
    
    This encoder uses Fourier harmonics to capture temporal patterns and modulate
    magnitude values based on time. It learns harmonic parameters that adapt to
    the time series characteristics.
    
    Args:
        num_harmonics (int): Number of Fourier harmonics to use. Default: 64
        embedding_size (int): Output embedding dimension. Default: 64
        Tmax (float): Maximum time value for normalization. Default: 3000.0
        input_size (int): Input feature dimension. Default: 1
        **kwargs: Additional keyword arguments
    """
    
    def __init__(self, num_harmonics=64, embedding_size=64, Tmax=3000.0, input_size=1, **kwargs):
        super(TimeFilm, self).__init__()
        
        # Initialize learnable Fourier parameters
        self.a = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        self.b = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        self.w = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        self.v = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        
        # Linear projections for magnitude encoding
        self.linear_proj = nn.Linear(in_features=input_size, out_features=embedding_size, bias=False)
        self.linear_proj_ = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=False)
        
        # Fixed time scaling parameter
        self.n_ = nn.Parameter(
            torch.linspace(1, num_harmonics + 1, steps=num_harmonics) / Tmax,
            requires_grad=False,
        )

        self.num_harmonics = num_harmonics

    def harmonics(self, t):
        """
        Generate harmonic frequencies for time values.
        
        Args:
            t: Time tensor of shape [batch, seq, 1]
            
        Returns:
            Harmonic time tensor of shape [batch, seq, 1, num_harmonics]
        """
        return t[:, :, None] * 2 * torch.pi * self.n_

    def fourier_coefs(self, t):
        """
        Compute Fourier coefficients for time modulation.
        
        Args:
            t: Time tensor of shape [batch, seq, 1]
            
        Returns:
            Tuple of (gamma, beta) modulation coefficients
        """
        t_harmonics = self.harmonics(t).float()
        
        # Calculate sin and cos once to avoid redundant computation
        sin_t = torch.sin(t_harmonics)
        cos_t = torch.cos(t_harmonics)
        
        gama_ = torch.tanh(
            torch.matmul(sin_t, self.a) + torch.matmul(cos_t, self.b)
        )
        
        beta_ = torch.matmul(sin_t, self.v) + torch.matmul(cos_t, self.w)
        
        return gama_, beta_

    def forward(self, x, t, band_info):
        """
        Forward pass applying time-dependent modulation to magnitude values.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            t: Time tensor of shape [batch, seq, 1]
            band_info: Band information tensor
            
        Returns:
            Time-modulated embeddings of shape [batch, seq, embedding_size]
        """
        gama_, beta_ = self.fourier_coefs(t)
        x = x.unsqueeze(-1)
        
        # Apply time-dependent modulation: x * gamma + beta
        x_proj = self.linear_proj(x)
        out = self.linear_proj_(x_proj * torch.squeeze(gama_) + torch.squeeze(beta_))
        
        return out
