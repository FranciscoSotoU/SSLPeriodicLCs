import torch
import torch.nn as nn
import copy

from .transformer import Transformer, Token
from .transformer.handlers import TimeHandler
from .utils import KSparse



class ATATLightcurve(nn.Module):
    """ATAT Lightcurve processor for handling time series lightcurve data."""
    
    def __init__(self, **kwargs):
        super(ATATLightcurve, self).__init__()
        
        self.num_bands = kwargs.get("num_bands", 2)
        
        # Setup lightcurve-specific kwargs

        # Initialize lightcurve components
        self.transformer_lc = Transformer(**kwargs)
        self.token_lc = Token(**kwargs)
        self.time_encoder = TimeHandler(**kwargs)
        self.k_sparse = KSparse(**kwargs)
        self.token_mode = kwargs.get("token_mode", None)
        self.init_model()

    def init_model(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)

    def add_token(self, x, mask=None, token=None):
        """Add token to the beginning of sequence."""
        batch_size, seq_len, dim = x.size()
        x = torch.cat([token, x], dim=1)
        m_token = torch.ones(batch_size, 1, 1).to(x.device)
        m = torch.cat([m_token, mask], dim=1)
        return x, m

    def _process_lc(self, x_mod, m_mod):
        """Process lightcurve data with token."""
        batch_size, _, _ = x_mod.size()
        x_, m_ = self.add_token(x=x_mod, mask=m_mod, token=self.token_lc(batch_size))
        return x_, m_

    def forward(self, data, time, mask=None, bands=None, **kwargs):
        """
        Forward pass for lightcurve processing.
        
        Args:
            data: Lightcurve flux data
            time: Time stamps
            mask: Mask for valid data points
            bands: Band information
            
        Returns:
            token_lc: Processed lightcurve token
        """
        # Encode time series data
        x_mod, m_mod, _ = self.time_encoder(x=data, t=time, mask=mask, band_info=bands)
        
        # Add token and process
        x_mod_token, m_mod_token = self._process_lc(x_mod, m_mod) if self.token_mode!='mean' else (x_mod, m_mod)
        
        # Transform through transformer
        token_lc = self.transformer_lc(
            x=x_mod_token,
            mask=m_mod_token
        )
        if self.token_mode == "mean":

            valid_mask = m_mod_token.squeeze(-1)  # Remove last dimension if present
            token_lc = token_lc * valid_mask.unsqueeze(-1)  # Zero out masked elements
            token_lc = token_lc.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)

        else:
            token_lc = token_lc[:, 0, :]

        if self.k_sparse is not None:
            token_lc = self.k_sparse(token_lc)
        return token_lc

    def get_attn_scores(self, data, time, mask=None, bands=None, **kwargs):
        """
        Get attention scores for lightcurve data.
        
        Args:
            data: Lightcurve flux data
            time: Time stamps
            mask: Mask for valid data points
            bands: Band information
            
        Returns:
            attn_lc: Attention scores
        """
        x_mod, m_mod, _ = self.time_encoder(x=data, t=time, mask=mask, band_info=bands)
        
        x_mod_token, m_mod_token = self._process_lc(x_mod, m_mod)

        _, attn_lc = self.transformer_lc(
            x=x_mod_token,
            mask=m_mod_token,
            get_attn=True
        )
        return attn_lc
