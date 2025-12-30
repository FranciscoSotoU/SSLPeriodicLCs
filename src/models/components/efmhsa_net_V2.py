import torch
import torch.nn as nn
import copy

from .transformer import Transformer, Token, Embedding
from .transformer.handlers import TimeHandler
from .utils import KSparse



class EFATAT(nn.Module):
    """EFATAT Lightcurve processor for handling time series lightcurve data."""
    
    def __init__(self,
                 time_encoder: nn.Module,
                 **kwargs):
        super(EFATAT, self).__init__()
        
        self.num_bands = kwargs.get("num_bands", 2)
        
        # Setup lightcurve-specific kwargs

        # Initialize lightcurve components
        self.transformer_lc = Transformer(**kwargs)
        self.token = Token(**kwargs)
        self.time_encoder = time_encoder
        self.k_sparse = KSparse(**kwargs)
        self.token_mode = kwargs.get("token_mode", None)
        
        #Features
                
        self.use_metadata = kwargs.get("use_metadata", False)
        self.use_features = kwargs.get("use_features", False)
        num_metadata = kwargs.get("num_metadata") if self.use_metadata else 0
        num_features = kwargs.get("num_features") if self.use_features else 0

        kwargs['length_size'] = num_metadata + num_features
        self.embedding_feat = Embedding(**kwargs)
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

    def _add_token(self, x_mod, m_mod):
        """Process lightcurve data with token."""
        batch_size, _, _ = x_mod.size()
        x_, m_ = self.add_token(x=x_mod, mask=m_mod, token=self.token(batch_size))
        return x_, m_
    def _process_metadata(self, metadata=None, features=None, period=None):
        """
        Process metadata, features, and period data.
        
        Args:
            metadata: Metadata features
            features: Additional features
            period: Period information
            
        Returns:
            tabular_feat: Processed tabular features
            tabular_mask: Corresponding mask
        """
        tabular_ = []
        
        if metadata is not None and self.use_metadata:
            tabular_.append(metadata)
            
        if features is not None and self.use_features:
            tabular_.append(features)
            
        if len(tabular_) > 0:
            tabular_feat = torch.cat(tabular_, dim=1).unsqueeze(-1)
        else:
            tabular_feat = tabular_[0].unsqueeze(-1)
            
        batch_size, _, _ = tabular_feat.size()
        tabular_mask = torch.ones_like(tabular_feat)
        tabular_feat = self.embedding_feat(tabular_feat)
        return tabular_feat, tabular_mask

    def forward(self, data, time, mask=None, bands=None, metadata=None, features=None, period=None, **kwargs):
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
        x_mod_lc, m_mod_lc, _ = self.time_encoder(x=data, t=time, mask=mask, band_info=bands)
        x_mod_feat, m_mod_feat = self._process_metadata(
            metadata=metadata, 
            features=features, 
        )
        # Add token and process
        x_mod = torch.cat([x_mod_lc, x_mod_feat], dim=1)
        m_mod = torch.cat([m_mod_lc, m_mod_feat], dim=1)
        
        x_mod_token, m_mod_token = self._add_token(x_mod, m_mod) if self.token_mode!='mean' else (x_mod, m_mod)
        
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
