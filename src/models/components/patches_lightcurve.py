import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_vit import Transformer, Token
from .utils import KSparse



class PatchedLightcurve(nn.Module):
    """ATAT Lightcurve processor for handling time series lightcurve data."""
    
    def __init__(self,
                 time_encoder: nn.Module,
                 **kwargs):
        super(PatchedLightcurve, self).__init__()
        
        self.num_bands = kwargs.get("num_bands", 2)
        self.patch_size = kwargs.get("patch_size", 8)  # Default patch size
        self.embedding_size = kwargs.get("embedding_size", 256)
        self.eps = 1e-8
        self.transformer_lc = Transformer(**kwargs)
        self.token_lc = Token(**kwargs)
        self.time_encoder = time_encoder
        self.k_sparse = KSparse(**kwargs)
        self.token_mode = kwargs.get("token_mode", None)
        self.proj = nn.Sequential(
            nn.Linear(self.patch_size, self.patch_size//2),
            nn.GELU(),
            nn.Linear(self.patch_size//2, 1),
        )
        self.proj_norm = nn.LayerNorm(self.embedding_size)  # Normalize the output of projection
        self.emb_do = nn.Dropout(kwargs.get("emb_dropout", 0.2 ))
        self.init_model()

    def init_model(self):
        """Initialize model parameters."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'conv' in name.lower():
                    nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(p, 0, 0.02)



    def add_token(self, x, mask=None, token=None):
        """Add token to the beginning of sequence."""
        batch_size, seq_len, dim = x.size()
        x = torch.cat([token, x], dim=1)
        m_token = torch.ones(batch_size, 1).to(x.device)  # Fixed: (B, 1) instead of (B, 1, 1)
        m = torch.cat([m_token, mask], dim=1)
        return x, m

    def _process_lc(self, x_mod, m_mod):
        """Process lightcurve data with token."""
        batch_size, _, _ = x_mod.size()
        x_, m_ = self.add_token(x=x_mod, mask=m_mod, token=self.token_lc(batch_size))
        return x_, m_


    def forward(self, data, time, mask=None, bands=None, patch_masks=None, **kwargs):
        """
        Forward pass for lightcurve processing with patching.
        
        Args:
            data: Lightcurve flux data
            time: Time stamps
            mask: Mask for valid data points
            bands: Band information
            
        Returns:
            token_lc: Processed lightcurve token
        """
        batch,seq_len,patch_len = data.shape
        data = data.view(batch, seq_len * patch_len)
        time = time.view(batch, seq_len * patch_len)
        mask = mask.view(batch, seq_len * patch_len)
        bands = bands.view(batch, seq_len * patch_len)
        x_mod, m_mod, _ = self.time_encoder(x=data, t=time, mask=mask, band_info=bands)
        x_mod = x_mod.view(batch, seq_len, patch_len, -1)
        m_mod = m_mod.view(batch, seq_len, patch_len,-1)
        

        masked_x = self.proj_norm(x_mod).permute(0,1,3,2)
        m_mod = m_mod.permute(0,1,3,2)
        masked_x = masked_x * m_mod
        masked_x = self.proj(masked_x).squeeze(-1)
    
        m_mod = m_mod.max(-1)[0].squeeze(-1)
        x_mod_token, m_mod_token = self._process_lc(masked_x, m_mod)
        x_mod_token = self.emb_do(x_mod_token)
        token_lc = self.transformer_lc(
            x=x_mod_token,
            mask=m_mod_token
        )
        token_lc = token_lc[:, 0, :]

        token_lc = self.k_sparse(token_lc)
        return token_lc
