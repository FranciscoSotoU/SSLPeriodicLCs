import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TimeHandler(nn.Module):
    def __init__(self, num_bands=2, embedding_size=64, Tmax=3000.0, **kwargs):
        super(TimeHandler, self).__init__()
        # general params
        time_encoder = kwargs.get("time_encoder", "TimeFilm")

        self.num_bands = num_bands
        self.embedding_size = embedding_size  # Fixed typo: ebedding_size -> embedding_size
        self.T_max = Tmax
        self.time_encoders = nn.ModuleList([
                            TimeFilm(embedding_size=embedding_size, Tmax=Tmax, **kwargs) for _ in range(num_bands)])

    def forward(self, x, t, mask=None, band_info=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        batch, seq_len = x.shape 
        # collect per‐band outputs
        x_mod = torch.zeros(batch, seq_len, self.embedding_size, device=x.device)
        
        for i in range(1, self.num_bands + 1):
            # Create band mask expanded to match dimensions
            band_mask = (band_info == i).unsqueeze(-1).expand(-1, -1, self.embedding_size)
            x_i = torch.where(band_info == i, x, torch.zeros_like(x))
            t_i = torch.where(band_info == i, t, torch.zeros_like(t))
            mod = self.time_encoders[i-1](x_i, t_i)  # Note: i-1 since ModuleList is 0-indexed
            
            # Use the expanded mask for element-wise selection
            x_mod = torch.where(band_mask, mod, x_mod)
        mask = mask.view(batch, seq_len, 1)
        t = t.view(batch, seq_len, 1)

        return x_mod, mask, t


class TimeFilm(nn.Module):
    def __init__(
        self, num_harmonics=64, embedding_size=64, Tmax=2500.0, input_size=1, **kwargs
    ):
        super(TimeFilm, self).__init__()
        # Initialize parameters more efficiently
        self.a = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        self.b = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        self.w = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        self.v = nn.Parameter(torch.rand(num_harmonics, embedding_size))
        
        # Simplify linear projections by removing unnecessary Sequential wrapper
        self.linear_proj = nn.Linear(in_features=input_size, out_features=embedding_size, bias=True)
        #self.linear_proj_ = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=True)

        # Fixed time scaling parameter
        self.n_ = nn.Parameter(
            torch.linspace(1, num_harmonics + 1, steps=num_harmonics) / Tmax,
            requires_grad=False,
        )

        self.num_harmonics = num_harmonics

    def harmonics(self, t):
        """t [n_batch, length sequence, 1, n_harmonics]"""
        # More efficient way to broadcast
    
        return t[:, :, None] * 2 * torch.pi * self.n_

    def fourier_coefs(self, t):
        t_harmonics = self.harmonics(t).float()
        
        # Calculate sin and cos once to avoid redundant computation
        sin_t = torch.sin(t_harmonics)
        cos_t = torch.cos(t_harmonics)
        
        gama_ = torch.matmul(sin_t, self.a) + torch.matmul(cos_t, self.b)
        beta_ = torch.matmul(sin_t, self.v) + torch.matmul(cos_t, self.w)
        
        return gama_, beta_

    def forward(self, x, t):
        """x [batch, seq, 1] , t [batch, seq, 1]"""
        gama_, beta_ = self.fourier_coefs(t)
        x = x.unsqueeze(-1)
        # Apply time-dependent modulation
        x_proj = self.linear_proj(x)
        out = x_proj * torch.squeeze(gama_) + torch.squeeze(beta_)
        #out = self.linear_proj_(x_proj * torch.squeeze(gama_) + torch.squeeze(beta_))
        
        return out