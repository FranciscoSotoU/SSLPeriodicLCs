import torch
import torch.nn as nn
import copy

from .transformer import Transformer, Token
from .transformer.handlers import TimeHandler
from .utils import KSparse



class ATATLightcurve(nn.Module):
    """ATAT Lightcurve processor for handling time series lightcurve data."""
    
    def __init__(self,
                 time_encoder: nn.Module,
                 **kwargs):
        super(ATATLightcurve, self).__init__()
        
        self.num_bands = kwargs.get("num_bands", 2)
        
        # Setup lightcurve-specific kwargs

        # Initialize lightcurve components
        self.transformer_lc = Transformer(**kwargs)
        self.token_lc = Token(**kwargs)
        self.time_encoder = time_encoder
        self.k_sparse = KSparse(**kwargs)
        self.token_mode = kwargs.get("token_mode", None)
        self.init_model()

    def init_model(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)


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
        

        # Transform through transformer


        valid_mask = m_mod.squeeze(-1)  # Remove last dimension if present
        token_lc = x_mod * valid_mask.unsqueeze(-1)  # Zero out masked elements
        token_lc = token_lc.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
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
