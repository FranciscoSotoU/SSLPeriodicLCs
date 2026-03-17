"""
Flexible Positional Encoder Module

This module contains the FlexiblePositionalEncoder class which allows switching
between different embedding strategies based on model parameters and configuration.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List


class FlexiblePositionalEncoder(nn.Module):
    """
    Flexible positional encoder that can adapt its embedding strategy based on parameters.
    
    This encoder supports multiple encoding strategies and can be configured to use
    different combinations of embeddings based on the model requirements:
    - Basic: Simple absolute embeddings
    - Enhanced: Time/magnitude differences and rates  
    - Advanced: Full feature set with multiple representations
    - Custom: User-defined combination of features
    
    Args:
        embedding_size (int): Output embedding dimension
        input_size (int): Input feature dimension. Default: 1
        seq_length (int): Maximum sequence length. Default: 2048
        num_bands (int): Number of photometric bands. Default: 2
        encoding_strategy (str): Strategy to use ('basic', 'enhanced', 'advanced', 'custom'). Default: 'enhanced'
        features_config (Dict): Configuration for which features to include. Default: None
        reduced_size_factor (int): Factor to reduce embedding dimensions for memory efficiency. Default: 2
        dropout (float): Dropout probability for regularization. Default: 0.1
        use_sinusoidal (bool): Whether to use sinusoidal time encoding. Default: True
        use_conv_mag (bool): Whether to use convolutional magnitude embedding. Default: True
        use_conv_time (bool): Whether to use convolutional time embedding. Default: False
        use_mag_diff (bool): Whether to include magnitude difference embeddings. Default: True
        use_time_diff (bool): Whether to include time difference embeddings. Default: True
        use_conv_time_diff (bool): Whether to use convolutional time difference embedding. Default: False
        use_rate (bool): Whether to include rate of change. Default: True
        use_conv_rate (bool): Whether to use convolutional rate embedding. Default: False
        use_band_embedding (bool): Whether to include band information. Default: True
        use_abs_time_mlp (bool): Whether to include absolute time MLP embedding. Default: None (auto)
        use_abs_mag_mlp (bool): Whether to include absolute magnitude MLP embedding. Default: None (auto)
        kernel_size (int): Kernel size for all convolutional layers. Default: 5
        fusion_strategy (str): How to combine embeddings ('mlp', 'attention', 'simple'). Default: 'mlp'
        mlp_layers (int): Number of hidden layers in MLP fusion. Default: 2
        **kwargs: Additional keyword arguments
    """
    
    def __init__(self, 
                 embedding_size: int, 
                 seq_length: int = 2048, 
                 num_bands: int = 2,
                 reduced_size_factor: int = 1,
                 dropout: float = 0.1,
                 use_sinusoidal: bool = True,
                 use_conv_mag: bool = True,
                 use_conv_time: bool = False,
                 use_mag_diff: bool = True,
                 use_time_diff: bool = True,
                 use_conv_time_diff: bool = False,
                 use_conv_mag_diff: bool = False,
                 use_rate: bool = True,                              
                 use_conv_rate: bool = False,
                 use_band_embedding: bool = True,
                 use_abs_time_mlp: bool = True,
                 use_abs_mag_mlp: bool = True,
                 kernel_size: int = 5,
                 fusion_strategy: str = 'simple',
                 mlp_layers: int = 3,
                 normalization_type: str = 'none',  # 'none', 'rms', 'layer'
                 normalize_embeddings: bool = False,  # Apply normalization after each embedding
                 **kwargs):
        super(FlexiblePositionalEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.seq_length = seq_length
        self.dropout = dropout
        self.fusion_strategy = fusion_strategy
        self.mlp_layers = mlp_layers
        self.num_bands = num_bands
        self.normalization_type = normalization_type
        self.normalize_embeddings = normalize_embeddings
        self.kernel_size = kernel_size
        self.use_conv_rate = use_conv_rate
        self.use_conv_time = use_conv_time
        self.use_conv_time_diff = use_conv_time_diff
        self.use_conv_mag_diff = use_conv_mag_diff
        
        # Configure features based on strategy
        self.features = self._configure_features(
            use_sinusoidal, use_conv_mag, use_conv_time, use_time_diff,
            use_mag_diff, use_rate, use_conv_rate, use_band_embedding, use_abs_time_mlp, use_abs_mag_mlp
        )
        
        # Calculate embedding dimensions
        reduced_size = int(embedding_size // reduced_size_factor)
        self.reduced_size = reduced_size
        intermediate_size = int(reduced_size // 2)
        conv_out_size = int(embedding_size // reduced_size_factor)
        
        # Initialize embedding components based on features
        self._init_embedding_components(
            reduced_size, intermediate_size, conv_out_size, num_bands
        )
        
        # Calculate total input size for fusion
        total_input_size = self._calculate_total_input_size(reduced_size, conv_out_size)
        
        # Initialize fusion layer
        self._init_fusion_layer(total_input_size, reduced_size)
        
        # Initialize normalization layers if needed
        self._init_normalization_layers()
        
        # Small epsilon to avoid division by zero
        self.eps = 1e-8
    
    def _configure_features(self,
                          use_sinusoidal: bool, use_conv_mag: bool, use_conv_time: bool, use_time_diff: bool,
                          use_mag_diff: bool, use_rate: bool, use_conv_rate: bool, use_band_embedding: bool,
                          use_abs_time_mlp: bool, use_abs_mag_mlp: bool) -> Dict[str, bool]:
        """Configure which features to use based on strategy."""
        
        features = {
            'sinusoidal_time': use_sinusoidal,
            'conv_magnitude': use_conv_mag,
            'conv_time': self.use_conv_time,
            'time_differences': use_time_diff,
            'conv_time_differences': self.use_conv_time_diff,
            'conv_mag_differences': self.use_conv_mag_diff,
            'mag_differences': use_mag_diff,
            'rate_of_change': use_rate,
            'conv_rate': use_conv_rate,
            'abs_time_mlp': use_abs_time_mlp ,
            'abs_mag_mlp': use_abs_mag_mlp,
            'band_embedding': use_band_embedding
        }

        # Ensure at least one embedding is enabled
        #return self._ensure_minimal_features(features)
        return features
    
    def _ensure_minimal_features(self, features: Dict[str, bool]) -> Dict[str, bool]:
        """Ensure at least one embedding is enabled. If none are enabled, enable abs_time_mlp and abs_mag_mlp."""
        # Check if any embedding is enabled (excluding band_embedding as it's optional)
        mag_embeddings = ['conv_magnitude', 'mag_differences', 'rate_of_change', 'abs_mag_mlp']
        time_embeddings = ['sinusoidal_time', 'time_differences', 'abs_time_mlp', 'rate_of_change']
        band_embeddings = ['band_embedding']
        any_mag_enabled = any(features.get(emb, False) for emb in mag_embeddings)
        any_time_enabled = any(features.get(emb, False) for emb in time_embeddings)
        if not any_mag_enabled:
            features['abs_mag_mlp'] = True
        if not any_time_enabled:
            features['sinusoidal_time'] = True

        return features
    
    def _init_embedding_components(self, reduced_size: int, intermediate_size: int,
                                 conv_out_size: int, num_bands: int):
        """Initialize embedding components based on features configuration."""
        
        if self.features['band_embedding']:
            self.band_embedding = nn.Embedding(
                num_embeddings=num_bands + 1, 
                embedding_dim=reduced_size
            )
        
        if self.features['time_differences']:
            self.time_diff_emb = nn.Linear(1, reduced_size)
        
        if self.features['conv_time_differences']:
            self.time_diff_conv_emb = nn.Conv1d(
                in_channels=1,
                out_channels=conv_out_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        
        if self.features['mag_differences']:
            self.mag_diff_emb = nn.Linear(1, reduced_size)

        if self.features['conv_mag_differences']:
            self.mag_diff_conv_emb = nn.Conv1d(
                in_channels=1,
                out_channels=conv_out_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        
        if self.features['rate_of_change']:
            self.rate_emb = nn.Linear(1, reduced_size)
        
        if self.features['conv_rate']:
            self.rate_conv_emb = nn.Conv1d(
                in_channels=1,
                out_channels=conv_out_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )

        if self.features['abs_time_mlp']:
            self.abs_time_emb = nn.Linear(1, reduced_size)
        
        if self.features['abs_mag_mlp']:
            self.abs_mag_emb = nn.Linear(1, reduced_size)
        
        if self.features['conv_magnitude']:
            self.magnitud_emb_big = nn.Conv1d(
                in_channels=1,
                out_channels=conv_out_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        
        if self.features['conv_time']:
            self.time_conv_emb_big = nn.Conv1d(
                in_channels=1,
                out_channels=conv_out_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        
        if self.features['sinusoidal_time'] or any(self.features[f] for f in ['conv_time_differences', 'conv_rate']):

            self.time_conv_emb = nn.Conv1d(
                in_channels=1,
                out_channels=conv_out_size,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
    
    def _calculate_total_input_size(self, reduced_size: int, conv_out_size: int) -> int:
        """Calculate total input size for fusion layer."""
        total_size = 0
        
        # Count embeddings with reduced_size
        for feature, enabled in self.features.items():
            if enabled and feature in ['time_differences', 'mag_differences', 'rate_of_change',
                                     'abs_time_mlp', 'abs_mag_mlp', 'band_embedding']:
                total_size += reduced_size
        
        # Add sinusoidal time embedding (reduced_size)
        if self.features['sinusoidal_time']:
            total_size += reduced_size
        
        # Add convolutional magnitude embedding (conv_out_size)
        if self.features['conv_magnitude']:
            total_size += conv_out_size
        
        # Add convolutional time embedding (conv_out_size)
        if self.features['conv_time']:
            total_size += conv_out_size
        
        # Add convolutional time differences embedding (conv_out_size)
        if self.features['conv_time_differences']:
            total_size += conv_out_size

        # Add convolutional magnitude differences embedding (conv_out_size)
        if self.features['conv_mag_differences']:
            total_size += conv_out_size
        
        # Add convolutional rate embedding (conv_out_size)
        if self.features['conv_rate']:
            total_size += conv_out_size
        
        return total_size
    
    def _init_fusion_layer(self, total_input_size: int, reduced_size: int):
        """Initialize fusion layer based on strategy."""
        
        if self.fusion_strategy == 'mlp':
            self.fusion_mlp = self._create_adaptive_mlp(total_input_size, reduced_size)
        elif self.fusion_strategy == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=total_input_size,
                num_heads=max(1, total_input_size // 64),
                dropout=self.dropout,
                batch_first=True
            )
            self.fusion_projection = nn.Linear(total_input_size, self.embedding_size)
            self.fusion_norm = nn.RMSNorm(self.embedding_size)
        elif self.fusion_strategy == 'simple':
            self.fusion_projection = nn.Linear(total_input_size, self.embedding_size)
            #self.fusion_norm = nn.RMSNorm(self.embedding_size)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _create_adaptive_mlp(self, input_size: int, reduced_size: int) -> nn.Sequential:
        """Create an adaptive MLP with configurable number of layers."""
        if self.mlp_layers < 1:
            raise ValueError("mlp_layers must be at least 1")
        
        layers = []
        
        # Initial activation and dropout
        layers.extend([
            nn.GELU(),
            nn.Dropout(self.dropout)
        ])
        
        # Calculate hidden sizes for each layer
        if self.mlp_layers == 1:
            # Single layer: input -> output
            layers.append(nn.Linear(input_size, self.embedding_size))
        else:
            # Multiple layers: input -> hidden -> ... -> hidden -> output
            fusion_hidden_size = reduced_size * 2
            
            # First hidden layer
            layers.extend([
                nn.Linear(input_size, fusion_hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            
            # Additional hidden layers (if mlp_layers > 2)
            for i in range(self.mlp_layers - 2):
                layers.extend([
                    nn.Linear(fusion_hidden_size, fusion_hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout) if i < self.mlp_layers - 3 else nn.Identity()  # No dropout before last layer
                ])
            
            # Final layer to output
            layers.extend([
                nn.Linear(fusion_hidden_size, self.embedding_size)
            ])
        

        return nn.Sequential(*layers)
    
    def compute_time_differences(self, t):
        """Compute time differences between consecutive steps."""
        batch_size, seq_len, _ = t.shape
        t_shifted = torch.cat([torch.zeros(batch_size, 1, 1, device=t.device), t[:, :-1]], dim=1)
        time_diffs = torch.abs(t - t_shifted)
        return time_diffs
    
    def compute_magnitude_differences(self, x):
        """Compute magnitude differences between consecutive steps."""
        batch_size, seq_len, _ = x.shape
        x_shifted = torch.cat([torch.zeros(batch_size, 1, 1, device=x.device), x[:, :-1]], dim=1)
        mag_diffs = x - x_shifted
        return mag_diffs
    
    def compute_rate(self, mag_diffs, time_diffs):
        """Compute rate of change: magnitude_diff / time_diff."""
        safe_time_diffs = time_diffs + self.eps
        rates = mag_diffs / safe_time_diffs
        rates = torch.clamp(rates, min=-1e15, max=1e15)
        return rates
    
    def sinusoidal_time_encoding(self, t, d_model):
        """Apply sinusoidal encoding to time values."""
        batch_size, seq_len, _ = t.shape
        div_term = torch.exp(torch.arange(0, d_model, 2, device=t.device) * 
                           -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(batch_size, seq_len, d_model, device=t.device)
        pe[:, :, 0::2] = torch.sin(t * div_term[None, None, :pe[:, :, 0::2].shape[-1]])
        
        if d_model % 2 == 1:
            pe[:, :, 1::2] = torch.cos(t * div_term[None, None, :pe[:, :, 1::2].shape[-1]])
        else:
            pe[:, :, 1::2] = torch.cos(t * div_term[None, None, :])
            
        return pe
    
    def temporal_embedding(self, t):
        """Enhanced temporal embedding with sinusoidal encoding."""
        return self.sinusoidal_time_encoding(t, self.reduced_size)
    
    def magnitud_embedding(self, x):
        """Encode magnitude values using 1D convolution."""
        x = x.transpose(1, 2)
        x = self.magnitud_emb_big(x)
        x = x.transpose(1, 2)
        return x
    

    
    def forward(self, x, t, bands=None):
        """
        Forward pass with flexible encoding strategy.
        
        Args:
            x: Magnitude tensor of shape [batch, seq, 1]
            t: Time tensor of shape [batch, seq, 1]
            bands: Band information (optional)
            
        Returns:
            Embeddings of shape [batch, seq, embedding_size]
        """
        # Ensure proper dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        embeddings = []
        
        # Compute differences if needed
        if any(self.features[f] for f in ['time_differences', 'conv_time_differences', 'rate_of_change', 'conv_rate']):
            time_diffs = self.compute_time_differences(t)
        if any(self.features[f] for f in ['mag_differences', 'conv_mag_differences', 'rate_of_change', 'conv_rate']):
            mag_diffs = self.compute_magnitude_differences(x)
        if self.features['rate_of_change'] or self.features['conv_rate']:
            rates = self.compute_rate(mag_diffs, time_diffs)
        
        # Generate embeddings based on configuration
        if self.features['sinusoidal_time']:
            t_emb = self.temporal_embedding(t)
            t_emb = self._apply_embedding_norm(t_emb, 'sinusoidal_time')
            embeddings.append(t_emb)
        
        if self.features['conv_magnitude']:
            # Normalize magnitudes to [0, 1] for stable conv
            x_emb = self.magnitud_embedding(x)
            x_emb = self._apply_embedding_norm(x_emb, 'conv_magnitude')
            embeddings.append(x_emb)
        
        if self.features['conv_time']:
            # Use convolutional embedding for time values
            t_conv = t.transpose(1, 2)
            t_conv = self.time_conv_emb_big(t_conv)
            t_conv = t_conv.transpose(1, 2)
            t_conv = self._apply_embedding_norm(t_conv, 'conv_time')
            embeddings.append(t_conv)
        
        if self.features['time_differences']:
            time_diff_emb = self.time_diff_emb(time_diffs)
            time_diff_emb = self._apply_embedding_norm(time_diff_emb, 'time_differences')
            embeddings.append(time_diff_emb)
        
        if self.features['conv_time_differences']:
            # Use convolutional embedding for time differences
            time_diff_conv = time_diffs.transpose(1, 2)
            time_diff_conv = self.time_diff_conv_emb(time_diff_conv)
            time_diff_conv = time_diff_conv.transpose(1, 2)
            time_diff_conv = self._apply_embedding_norm(time_diff_conv, 'conv_time_differences')
            embeddings.append(time_diff_conv)
        
        if self.features['mag_differences']:
            mag_diff_emb = self.mag_diff_emb(mag_diffs)
            mag_diff_emb = self._apply_embedding_norm(mag_diff_emb, 'mag_differences')
            embeddings.append(mag_diff_emb)

        if self.features['conv_mag_differences']:
            # Use convolutional embedding for magnitude differences
            mag_diff_normalized = self.normalize_values(mag_diffs)
            mag_diff_conv = mag_diff_normalized.transpose(1, 2)
            mag_diff_conv = self.mag_diff_conv_emb(mag_diff_conv)
            mag_diff_conv = mag_diff_conv.transpose(1, 2)
            mag_diff_conv = self._apply_embedding_norm(mag_diff_conv, 'conv_mag_differences')
            embeddings.append(mag_diff_conv)
        
        if self.features['rate_of_change']:
            rate_emb = self.rate_emb(rates)
            rate_emb = self._apply_embedding_norm(rate_emb, 'rate_of_change')
            embeddings.append(rate_emb)
        
        if self.features['conv_rate']:

            rate_conv = rates.transpose(1, 2)
            rate_conv = self.rate_conv_emb(rate_conv)
            rate_conv = rate_conv.transpose(1, 2)
            rate_conv = self._apply_embedding_norm(rate_conv, 'conv_rate')
            embeddings.append(rate_conv)

        if self.features['abs_time_mlp']:
            abs_time_emb = self.abs_time_emb(t)
            abs_time_emb = self._apply_embedding_norm(abs_time_emb, 'abs_time_mlp')
            embeddings.append(abs_time_emb)
        
        if self.features['abs_mag_mlp']:
            abs_mag_emb = self.abs_mag_emb(x)
            abs_mag_emb = self._apply_embedding_norm(abs_mag_emb, 'abs_mag_mlp')
            embeddings.append(abs_mag_emb)
        
        if self.features['band_embedding'] and bands is not None:
            band_emb = self.band_embedding(bands.int())
            band_emb = self._apply_embedding_norm(band_emb, 'band_embedding')
            embeddings.append(band_emb)
        
        # Combine embeddings
        if not embeddings:
            raise ValueError("No embeddings configured. Check your features configuration.")
        
        combined = torch.cat(embeddings, dim=2)
        
        # Apply fusion strategy
        if self.fusion_strategy == 'mlp':
            out = self.fusion_mlp(combined)
        elif self.fusion_strategy == 'attention':
            # Use self-attention for fusion
            attn_out, _ = self.fusion_attention(combined, combined, combined)
            out = self.fusion_projection(attn_out)
            out = self.fusion_norm(out)
        elif self.fusion_strategy == 'simple':
            out = self.fusion_projection(combined)
        return out
    
    def _init_normalization_layers(self):
        """Initialize normalization layers for embeddings if enabled."""
        if not self.normalize_embeddings or self.normalization_type == 'none':
            return
            
        # Create normalization layers for each embedding type
        self.norm_layers = nn.ModuleDict()
        
        if self.features['sinusoidal_time']:
            self.norm_layers['sinusoidal_time'] = self._create_norm_layer(self.reduced_size)
        
        if self.features['conv_magnitude']:
            conv_out_size = self.embedding_size // 2
            self.norm_layers['conv_magnitude'] = self._create_norm_layer(conv_out_size)
        
        if self.features['conv_time']:
            conv_out_size = self.embedding_size // 2
            self.norm_layers['conv_time'] = self._create_norm_layer(conv_out_size)
        
        if self.features['time_differences']:
            self.norm_layers['time_differences'] = self._create_norm_layer(self.reduced_size)
        
        if self.features['conv_time_differences']:
            conv_out_size = self.embedding_size // 2
            self.norm_layers['conv_time_differences'] = self._create_norm_layer(conv_out_size)
        
        if self.features['mag_differences']:
            self.norm_layers['mag_differences'] = self._create_norm_layer(self.reduced_size)
        
        if self.features['rate_of_change']:
            self.norm_layers['rate_of_change'] = self._create_norm_layer(self.reduced_size)
        
        if self.features['conv_rate']:
            conv_out_size = self.embedding_size // 2
            self.norm_layers['conv_rate'] = self._create_norm_layer(conv_out_size)
        
        if self.features['abs_time_mlp']:
            self.norm_layers['abs_time_mlp'] = self._create_norm_layer(self.reduced_size)
        
        if self.features['abs_mag_mlp']:
            self.norm_layers['abs_mag_mlp'] = self._create_norm_layer(self.reduced_size)
        
        if self.features['band_embedding']:
            self.norm_layers['band_embedding'] = self._create_norm_layer(self.reduced_size)
    
    def _create_norm_layer(self, dim: int):
        """Create a normalization layer based on the specified type."""
        if self.normalization_type == 'rms':
            return nn.RMSNorm(dim)
        elif self.normalization_type == 'layer':
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()
    
    def _apply_embedding_norm(self, embedding: torch.Tensor, embedding_type: str) -> torch.Tensor:
        """Apply normalization to an embedding if configured."""
        if (self.normalize_embeddings and 
            self.normalization_type != 'none' and 
            hasattr(self, 'norm_layers') and 
            embedding_type in self.norm_layers):
            return self.norm_layers[embedding_type](embedding)
        return embedding
    

class FlexiblePositionalEncoderHandler(nn.Module):
    """
    Configurable time handler for processing time series data.
    
    This class supports multiple encoder types and processes bands sequentially,
    providing flexibility in time series encoding strategies.
    """
    def __init__(self, 
                embedding_size: int, 
                seq_length: int = 2048, 
                num_bands: int = 2,
                reduced_size_factor: int = 2,
                dropout: float = 0.1,
                use_sinusoidal: bool = True,
                use_conv_mag: bool = True,
                use_conv_time: bool = False,
                use_mag_diff: bool = True,
                use_time_diff: bool = True,
                use_rate: bool = True,
                use_band_embedding: bool = True,
                use_abs_time_mlp: bool = True,
                use_abs_mag_mlp: bool = True,
                use_conv_rate: bool = True,
                use_conv_time_diff: bool = True,
                kernel_size: int = 5,
                fusion_strategy: str = 'simple',
                mlp_layers: int = 3,
                normalization_type: str = 'none',  # 'none', 'rms', 'layer'
                normalize_embeddings: bool = False,  # Apply normalization after each embedding
                **kwargs):
        super(FlexiblePositionalEncoderHandler, self).__init__()

        self.num_bands = num_bands
        self.embedding_size = embedding_size

        # Create time encoders with all the parameters
        encoder_kwargs = {
            'embedding_size': embedding_size,
            'seq_length': seq_length,
            'num_bands': num_bands,
            'reduced_size_factor': reduced_size_factor,
            'dropout': dropout,
            'use_sinusoidal': use_sinusoidal,
            'use_conv_mag': use_conv_mag,
            'use_conv_time': use_conv_time,
            'use_mag_diff': use_mag_diff,
            'use_time_diff': use_time_diff,
            'use_conv_time_diff': use_conv_time_diff,
            'use_rate': use_rate,
            'use_conv_rate': use_conv_rate,
            'use_band_embedding': use_band_embedding,
            'use_abs_time_mlp': use_abs_time_mlp,
            'use_abs_mag_mlp': use_abs_mag_mlp,
            'kernel_size': kernel_size,
            'fusion_strategy': fusion_strategy,
            'mlp_layers': mlp_layers,
            'normalization_type': normalization_type,
            'normalize_embeddings': normalize_embeddings,
            **kwargs
        }
        
        # Use nn.ModuleList for proper parameter registration
        self.time_encoders = nn.ModuleList([
            FlexiblePositionalEncoder(**encoder_kwargs) 
            for _ in range(num_bands)
        ])

    def forward(self, x, t, mask=None, band_info=None, **kwargs):
        """
        Forward pass for time handling with band processing.
        
        Args:
            x: Input magnitude tensor of shape [batch, seq]
            t: Time tensor of shape [batch, seq]
            mask: Optional mask tensor. If None, all positions are considered valid
            band_info: Band information tensor indicating which band each observation belongs to
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (x_mod, mask, t) where:
                - x_mod: Encoded tensor of shape [batch, seq, embedding_size]
                - mask: Reshaped mask tensor of shape [batch, seq, 1]
                - t: Reshaped time tensor of shape [batch, seq, 1]
        """
        batch, seq_len = x.shape

        # collect per‐band outputs
        x_mod = torch.zeros(batch, seq_len, self.embedding_size, device=x.device)
        
        for i in range(1, self.num_bands + 1):
            # Create band mask expanded to match dimensions
            band_mask = (band_info == i).unsqueeze(-1).expand(-1, -1, self.embedding_size)
            
            x_i = torch.where(band_info == i, x, torch.zeros_like(x))
            t_i = torch.where(band_info == i, t, torch.zeros_like(t))
            b_i = torch.where(band_info == i, band_info, torch.zeros_like(mask))

            # Sort by band information for processing
            index = torch.argsort(b_i, dim=1, descending=True, stable=True)
            x_i = torch.gather(x_i, 1, index)
            t_i = torch.gather(t_i, 1, index)
            b_i = torch.gather(b_i, 1, index)

            mod = self.time_encoders[i-1](x_i, t_i, b_i)  # Note: i-1 since ModuleList is 0-indexed
            
            # Inverse the index to get back to original order
            # Compute inverse permutation
            inverse_index = torch.zeros_like(index)
            inverse_index.scatter_(1, index, torch.arange(index.size(1), device=index.device).unsqueeze(0).expand_as(index))

            # Reorder mod back to original
            mod = torch.gather(mod, 1, inverse_index.unsqueeze(-1).expand(-1, -1, self.embedding_size))

            # Use the expanded mask for element-wise selection
            x_mod = torch.where(band_mask, mod, x_mod)
            
        mask = mask.view(batch, seq_len, 1)

        return x_mod, mask, band_info


