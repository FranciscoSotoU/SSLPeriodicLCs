import torch
import torch.nn as nn
import copy

from .transformer import Transformer, Token
from .transformer.handlers import TimeHandler
from .utils import KSparse



class ConvolutionalLightcurve(nn.Module):
    """Convolutional Lightcurve processor for handling time series lightcurve data."""
    
    def __init__(self, **kwargs):
        super(ConvolutionalLightcurve, self).__init__()
        
        self.num_bands = kwargs.get("num_bands", 2)
        
        # Setup lightcurve-specific kwargs

        # Initialize lightcurve components
        #self.transformer_lc = Transformer(**kwargs)
        #self.token_lc = Token(**kwargs)
        self.time_encoder = TimeHandler(**kwargs)
        self.k_sparse = KSparse(**kwargs)
        self.token_mode = kwargs.get("token_mode", None)
        self.init_model()
        self.lc_encoder = Inception_Block_V1(emb=kwargs.get("embedding_size", 128), emb_cls=kwargs.get("embedding_size", 128))

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
        #multiply x_mod with mask to make zero the masked values
        x_ = x_mod * m_mod
        return x_, m_mod

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
        x_mod_token, m_mod_token = self._process_lc(x_mod, m_mod)
        
        # Transform through transformer
        token_lc = self.lc_encoder(
            x=x_mod_token,
        )
        if self.k_sparse is not None:
            token_lc = self.k_sparse(token_lc)
        return token_lc




class TwoLevelConvPoolingAvg(nn.Module):
    def __init__(self, emb, emb_cls):
        super().__init__()
        
        # Level 1
        self.conv1 = nn.Conv1d(emb, emb_cls * 2, kernel_size=9, padding=4)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        # Level 2
        self.conv2 = nn.Conv1d(emb_cls * 2, emb_cls, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        
        # Final global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = x.transpose(1, 2)       # (batch, emb, seq_len)
        
        # Level 1
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Level 2
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        return x
    
class Inception_Block_V1(nn.Module):
    def __init__(self, emb, emb_cls, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = emb
        self.out_channels = emb_cls
        self.num_kernels = num_kernels
        
        # Level 1: Multiple parallel convolutions with different kernel sizes
        kernels = []
        for i in range(self.num_kernels):
            kernel_size = 2 * i + 3  # kernel sizes: 3, 5, 7, 9, 11, 13
            padding = kernel_size // 2
            kernels.append(nn.Conv1d(emb, emb_cls * 2, kernel_size=kernel_size, padding=padding))
        self.kernels = nn.ModuleList(kernels)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        
        # Level 2: Final convolution to match output dimensions
        self.conv2 = nn.Conv1d(emb_cls * 2, emb_cls, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        
        # Final global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch, seq, emb)
        x = x.transpose(1, 2)  # (batch, emb, seq_len)
        
        # Level 1: Apply multiple parallel convolutions
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(torch.relu(self.kernels[i](x)))
        
        # Combine results by averaging
        x = torch.stack(res_list, dim=-1).mean(-1)
        x = self.pool1(x)
        
        # Level 2
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Global pooling to get (batch, emb_cls)
        x = self.global_pool(x).squeeze(-1)
        
        return x  # Output: (batch, emb_cls)

