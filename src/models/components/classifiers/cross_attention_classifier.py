import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionClassifier(nn.Module):
    """
    Cross-attention classifier that fuses two modalities (e.g., lightcurve and tabular).
    
    Args:
        embedding_size: Dimension of embeddings from each modality
        num_classes: Number of output classes
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        **kwargs: Additional arguments (e.g., activation, norm)
    """
    def __init__(self, embedding_size, num_classes, num_heads=8, dropout=0.1, **kwargs):
        super(CrossAttentionClassifier, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        
        # Layer normalization for each modality
        self.norm_lc = nn.LayerNorm(embedding_size)
        self.norm_tab = nn.LayerNorm(embedding_size)
        
        # Cross-attention: lightcurve attends to tabular
        self.cross_attn_lc = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: tabular attends to lightcurve
        self.cross_attn_tab = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks for each modality
        ff_hidden = embedding_size * 4
        self.ff_lc = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embedding_size),
            nn.Dropout(dropout)
        )
        
        self.ff_tab = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embedding_size),
            nn.Dropout(dropout)
        )
        
        # Output normalization
        self.out_norm = nn.LayerNorm(embedding_size * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_classes)
        )
        
        self.init_model()
    
    def init_model(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, lc_cls, tab_embedding):
        """
        Forward pass with cross-attention fusion using CLS tokens.
        
        Args:
            lc_cls: Lightcurve CLS token [batch_size, embedding_size]
            tab_embedding: Tabular embedding [batch_size, embedding_size]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = lc_cls.shape[0]
        
        # Add sequence dimension for multi-head attention
        lc_tokens = lc_cls.unsqueeze(1)  # [batch_size, 1, embedding_size]
        tab_tokens = tab_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
        
        # Normalize inputs
        lc_norm = self.norm_lc(lc_tokens)
        tab_norm = self.norm_tab(tab_tokens)
        
        # Cross-attention: lc attends to tab
        lc_cross, _ = self.cross_attn_lc(
            query=lc_norm,
            key=tab_norm,
            value=tab_norm,
            need_weights=False
        )
        lc_cross = lc_tokens + lc_cross  # Residual connection
        
        # Cross-attention: tab attends to lc
        tab_cross, _ = self.cross_attn_tab(
            query=tab_norm,
            key=lc_norm,
            value=lc_norm,
            need_weights=False
        )
        tab_cross = tab_tokens + tab_cross  # Residual connection
        
        # Feed-forward networks
        lc_ff = self.ff_lc(lc_cross)
        lc_out = lc_cross + lc_ff  # Residual connection
        
        tab_ff = self.ff_tab(tab_cross)
        tab_out = tab_cross + tab_ff  # Residual connection
        
        # Remove sequence dimension and concatenate
        lc_final = lc_out.squeeze(1)  # [batch_size, embedding_size]
        tab_final = tab_out.squeeze(1)  # [batch_size, embedding_size]
        
        # Concatenate modalities
        fused = torch.cat([lc_final, tab_final], dim=1)  # [batch_size, embedding_size * 2]
        
        # Output normalization and classification
        fused_norm = self.out_norm(fused)
        logits = self.classifier(fused_norm)
        
        return logits
