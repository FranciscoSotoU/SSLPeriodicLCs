import torch
import torch.nn as nn
import copy

from .transformer import Transformer, Token, Embedding
from .utils import KSparse


class ATATFeatures(nn.Module):
    """ATAT Features processor for handling metadata, features, and period data."""
    
    def __init__(self, **kwargs):
        super(ATATFeatures, self).__init__()
        
        self.use_metadata = kwargs.get("use_metadata", False)
        self.use_features = kwargs.get("use_features", False)

        # Calculate input sizes
        num_metadata = kwargs.get("num_metadata") if self.use_metadata else 0
        num_features = kwargs.get("num_features") if self.use_features else 0
        
        # Setup features-specific kwargs

        kwargs['length_size'] = num_metadata + num_features
        
        # Initialize feature components
        self.transformer_feat = Transformer(**kwargs)
        self.token_feat = Token(**kwargs)
        self.embedding_feat = Embedding(**kwargs)
        self.k_sparse = KSparse(**kwargs)
        self.token_mode = kwargs.get("token_mode", None)
        
        
        self.init_model()

    def init_model(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)

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
        token = self.token_feat(batch_size)
        tabular_mask = torch.ones_like(tabular_feat)
        tabular_feat = self.embedding_feat(tabular_feat)
        tabular_feat = torch.cat([token, tabular_feat], dim=1)
        token_mask = torch.ones(batch_size, 1, 1).to(tabular_feat.device)
        tabular_mask = torch.cat([token_mask, tabular_mask], dim=1)
        
        return tabular_feat, tabular_mask

    def forward(self, metadata=None, features=None, period=None, **kwargs):
        """
        Forward pass for features processing.
        
        Args:
            metadata: Metadata features
            features: Additional features  
            period: Period information
            
        Returns:
            token_feat: Processed features token
        """
        x_mod_feat, m_mod_feat = self._process_metadata(
            metadata=metadata, 
            features=features, 
        )
        
        token_feat = self.transformer_feat(
            x=x_mod_feat,
            mask=m_mod_feat
        )
        if self.token_mode == "mean":
            # Calculate mean only for non-masked elements
            valid_mask = m_mod_feat.squeeze(-1)  # Remove last dimension if present
            token_feat = token_feat * valid_mask.unsqueeze(-1)  # Zero out masked elements
            token_feat = token_feat.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)

        else:
            token_feat = token_feat[:, 0, :]
        if self.token_mode == "k_sparse":
            if self.k_sparse is not None:
                token_feat = self.k_sparse(token_feat)
        return token_feat

    def get_attn_scores(self, metadata=None, features=None, period=None, **kwargs):
        """
        Get attention scores for features data.
        
        Args:
            metadata: Metadata features
            features: Additional features
            period: Period information
            
        Returns:
            attn_ft: Attention scores
        """
        x_mod_feat, m_mod_feat = self._process_metadata(
            metadata=metadata, 
            features=features,
            period=period
        )
        
        _, attn_ft = self.transformer_feat(
            x=x_mod_feat,
            mask=m_mod_feat,
            get_attn=True
        )
        
        return attn_ft

    def has_features_enabled(self):
        """Check if any feature type is enabled."""
        return self.use_metadata or self.use_features or self.use_period
