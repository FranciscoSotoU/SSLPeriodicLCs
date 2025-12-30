import torch
import torch.nn as nn
import copy

from .transformer import Transformer, TimeHandlerParallel, Harmonics, Token, Embedding, TimeHandler
from .utils import KSparse


class EFMHSA(nn.Module):
    """
    Enhanced Feature Multi-Head Self-Attention network for light curve analysis.
    
    Supports various input modalities:
    - Time series data with temporal encoding
    - Metadata features  
    - Periodogram data
    - Period information with harmonics
    """
    
    def __init__(self, **kwargs):
        super(EFMHSA, self).__init__()
        
        # Setup configuration and encoders
        self._setup_config(kwargs)
        self._setup_encoders(kwargs)
        self._setup_transformer(kwargs)
        
        # Initialize model weights
        self.init_model()
    
    def _setup_config(self, kwargs):
        """Extract and store configuration parameters."""
        # Feature flags
        self.use_metadata = kwargs.get("use_metadata", False)
        self.use_features = kwargs.get("use_features", False)
        self.use_period = kwargs.get("use_period", False)
        self.use_periodogram = kwargs.get("use_periodogram", False)
        self.periodogram_linear = kwargs.get("periodogram_linear", True)
        
        # Architecture settings
        self.time_handler = kwargs.get("time_handler", 'Sequential')
        self.use_k_sparse = kwargs.get("ksparse", 0)
        
        # Dimensions
        self.num_metadata = kwargs.get("num_metadata", 0)
        self.num_features = kwargs.get("num_features", 0)
        self.num_periodogram = kwargs.get("num_periodogram", 10)
        self.num_periods_periodogram = kwargs.get("num_periods_periodogram", 1)
        self.num_bands = kwargs.get("num_bands", 2)
        
        # Random masking configuration
        self.use_random_masking = kwargs.get("use_random_masking", False)
        self.random_mask_prob = kwargs.get("random_mask_prob", 0.15)
        self.mask_tabular_during_training = kwargs.get("mask_tabular_during_training", False)
    
    def _setup_encoders(self, kwargs):
        """Initialize encoder modules based on configuration."""
        # Time encoder - core component for temporal processing
        if self.time_handler == 'Parallel':
            self.time_encoder = TimeHandlerParallel(**kwargs)
        else:
            self.time_encoder = TimeHandler(**kwargs)
        
        # Optional feature encoders
        if self.use_period:
            self.period_encoder = Embedding(length_size=1, **kwargs)
            self.harmonics = Harmonics(**kwargs)
            
        if self.use_metadata:
            self.metadata_encoder = Embedding(length_size=self.num_metadata, **kwargs)
            
        if self.use_features:
            self.feature_encoder = Embedding(length_size=self.num_features, **kwargs)
            
        if self.use_periodogram:
            print("Initializing periodogram encoder")
            periodogram_size = self.num_periodogram * self.num_periods_periodogram
            if self.periodogram_linear:
                self.periodogram_encoder = nn.Linear(
                    in_features=self.num_periodogram, out_features=kwargs['embedding_size'], bias=True
                )
            else:
                self.periodogram_encoder = Embedding(length_size=periodogram_size, **kwargs)
                self.harmonics_periodogram = Harmonics(num_periods=self.num_periods_periodogram)
    
    def _setup_transformer(self, kwargs):
        """Initialize transformer and output components."""
        # Main transformer for sequence processing
        self.transformer_lc = Transformer(**kwargs)
        
        # Optional k-sparse attention
        if self.use_k_sparse > 0:
            self.k_sparse = KSparse(percentage=self.use_k_sparse)
        
        # Band-specific tokens
        token_template = Token(**kwargs)
        self.token_lc = nn.ModuleList([
            copy.deepcopy(token_template) for _ in range(self.num_bands)
        ])

    def init_model(self):
        """Initialize model weights with normal distribution."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def _random_zero_mask_tabular(self, x, mask, start_idx, end_idx, mask_prob=0.15):
        """
        Randomly zero mask tabular data positions.
        
        Args:
            x: Input tensor [batch, seq_len, embedding_dim]
            mask: Attention mask [batch, seq_len, 1] 
            start_idx: Start index of tabular features in sequence
            end_idx: End index of tabular features in sequence
            mask_prob: Probability of masking each position
            
        Returns:
            x_masked: Input tensor with randomly masked tabular features
            mask_masked: Updated attention mask
        """
        if not self.training or start_idx >= end_idx:
            return x, mask
            
        batch_size = x.shape[0]
        num_tabular = end_idx - start_idx
        device = x.device
        
        # Generate random mask for tabular positions
        random_mask = torch.rand(batch_size, num_tabular, device=device) > mask_prob
        
        # Create a copy to avoid in-place operations
        x_masked = x.clone()
        mask_masked = mask.clone() if mask is not None else None
        
        # Apply random masking to tabular features
        x_masked[:, start_idx:end_idx] = x_masked[:, start_idx:end_idx] * random_mask.unsqueeze(-1)
        
        # Update attention mask for masked positions
        if mask_masked is not None:
            attention_mask = random_mask.unsqueeze(-1)  # [batch, num_tabular, 1]
            mask_masked[:, start_idx:end_idx] = mask_masked[:, start_idx:end_idx] * attention_mask
        
        return x_masked, mask_masked

    def _apply_random_tabular_masking(self, x, mask, metadata=None, features=None, 
                                    period=None, periodogram=None):
        """
        Apply random masking to all types of tabular data.
        
        Args:
            x: Input tensor [batch, seq_len, embedding_dim]
            mask: Attention mask [batch, seq_len, 1]
            metadata: Metadata features (for counting)
            features: Feature data (for counting)
            period: Period data (for counting) 
            periodogram: Periodogram data (for counting)
            
        Returns:
            x_masked: Input tensor with randomly masked tabular features
            mask_masked: Updated attention mask
        """
        if not self.use_random_masking or not self.training:
            return x, mask
        current_idx = 1  # Start after token
        x_masked, mask_masked = x, mask
        
        # Mask metadata positions
        if self.use_metadata and metadata is not None:
            end_idx = current_idx + self.num_metadata
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, self.random_mask_prob
            )
            current_idx = end_idx
        
        # Mask feature positions  
        if self.use_features and features is not None:
            end_idx = current_idx + self.num_features
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, self.random_mask_prob
            )
            current_idx = end_idx
        
        # Mask period positions
        if self.use_period and period is not None:
            num_harmonics = getattr(self.harmonics, 'num_periods', 1) if hasattr(self, 'harmonics') else 1
            end_idx = current_idx + num_harmonics
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, self.random_mask_prob
            )
            current_idx = end_idx
        
        # Mask periodogram positions
        if self.use_periodogram and periodogram is not None:
            num_periodogram_features = self.num_periodogram * self.num_periods_periodogram
            end_idx = current_idx + num_periodogram_features
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, self.random_mask_prob
            )
            current_idx = end_idx
        
        return x_masked, mask_masked

    def _selective_random_mask_tabular(self, x, mask, metadata=None, features=None,
                                     period=None, periodogram=None,
                                     mask_metadata=True, mask_features=True,
                                     mask_period=True, mask_periodogram=True,
                                     mask_prob=None):
        """
        Selectively apply random masking to specific types of tabular data.
        
        Args:
            x: Input tensor [batch, seq_len, embedding_dim]
            mask: Attention mask [batch, seq_len, 1]
            metadata, features, period, periodogram: Input data for counting
            mask_metadata, mask_features, mask_period, mask_periodogram: Flags for selective masking
            mask_prob: Override probability (uses self.random_mask_prob if None)
            
        Returns:
            x_masked: Input tensor with selectively masked tabular features
            mask_masked: Updated attention mask
        """
        if not self.training:
            return x, mask
            
        prob = mask_prob if mask_prob is not None else self.random_mask_prob
        current_idx = 1  # Start after token
        x_masked, mask_masked = x, mask
        
        # Selectively mask each type of tabular data
        if self.use_metadata and metadata is not None and mask_metadata:
            end_idx = current_idx + self.num_metadata
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, prob
            )
        if self.use_metadata and metadata is not None:
            current_idx += self.num_metadata
        
        if self.use_features and features is not None and mask_features:
            end_idx = current_idx + self.num_features
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, prob
            )
        if self.use_features and features is not None:
            current_idx += self.num_features
        
        if self.use_period and period is not None and mask_period:
            num_harmonics = getattr(self.harmonics, 'num_periods', 1) if hasattr(self, 'harmonics') else 1
            end_idx = current_idx + num_harmonics
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, prob
            )
        if self.use_period and period is not None:
            num_harmonics = getattr(self.harmonics, 'num_periods', 1) if hasattr(self, 'harmonics') else 1
            current_idx += num_harmonics
        
        if self.use_periodogram and periodogram is not None and mask_periodogram:
            num_periodogram_features = self.num_periodogram * self.num_periods_periodogram
            end_idx = current_idx + num_periodogram_features
            x_masked, mask_masked = self._random_zero_mask_tabular(
                x_masked, mask_masked, current_idx, end_idx, prob
            )
        
        return x_masked, mask_masked

    def _add_metadata(self, x, mask, metadata):
        """Add metadata features to the sequence."""
        if not self.use_metadata or metadata is None:
            return x, mask
            
        batch_size, device, dtype = x.shape[0], x.device, x.dtype
        
        # Encode metadata
        metadata_encoded = self.metadata_encoder(metadata.unsqueeze(-1)).to(dtype=dtype)
        x = torch.cat([metadata_encoded, x], dim=1)
        
        # Update mask
        if mask is not None:
            metadata_mask = torch.ones(
                batch_size, self.num_metadata, 1, 
                device=device, dtype=dtype
            )
            mask = torch.cat([metadata_mask, mask], dim=1)
            
        return x, mask
    
    def _add_features(self, x, mask, features):
        """Add engineered features to the sequence."""
        if not self.use_features or features is None:
            return x, mask
            
        batch_size, device, dtype = x.shape[0], x.device, x.dtype
        
        # Encode features
        features_encoded = self.feature_encoder(features.unsqueeze(-1)).to(dtype=dtype)
        x = torch.cat([features_encoded, x], dim=1)
        
        # Update mask
        if mask is not None:
            features_mask = torch.ones(
                batch_size, self.num_features, 1,
                device=device, dtype=dtype
            )
            mask = torch.cat([features_mask, mask], dim=1)
            
        return x, mask
    
    def _add_period(self, x, mask, period):
        """Add period information with harmonic encoding."""
        if not self.use_period or period is None:
            return x, mask
            
        batch_size, device, dtype = x.shape[0], x.device, x.dtype
        
        # Generate harmonic periods
        harmonics = self.harmonics()
        period_expanded = period.view(-1, 1)
        harmonic_periods = (harmonics * period_expanded).unsqueeze(-1)
        
        # Encode periods
        period_encoded = self.period_encoder(harmonic_periods).to(dtype=dtype)
        x = torch.cat([period_encoded, x], dim=1)
        # Update mask
        if mask is not None:
            period_mask = torch.ones(
                batch_size, harmonic_periods.shape[1], 1,
                device=device, dtype=dtype
            )
            mask = torch.cat([period_mask, mask], dim=1)
            
        return x, mask
    
    def _add_periodogram(self, x, mask, periodogram):
        """Add periodogram features to the sequence."""
        if not self.use_periodogram or periodogram is None:
            return x, mask
            
        batch_size, device, dtype = x.shape[0], x.device, x.dtype
        
        # Process periodogram
        if self.periodogram_linear:
            periodogram_encoded = self.periodogram_encoder(periodogram[:, :self.num_periodogram])
            periodogram_encoded = periodogram_encoded.unsqueeze(1)
            x = torch.cat([periodogram_encoded, x], dim=1)
            # Update mask
            if mask is not None:
                periodogram_mask = torch.ones(
                    batch_size, 1, 1,
                    device=device, dtype=dtype
                )
                mask = torch.cat([periodogram_mask, mask], dim=1)
        else:
            periodogram_truncated = periodogram[:, :self.num_periodogram].unsqueeze(-1)
            periodogram_harmonics = self.harmonics_periodogram() * periodogram_truncated 
            # Reshape and encode
            periodogram_flat = periodogram_harmonics.reshape(
                batch_size, self.num_periodogram * self.num_periods_periodogram, 1
            )
            periodogram_encoded = self.periodogram_encoder(periodogram_flat).to(dtype=dtype)
        
            x = torch.cat([periodogram_encoded, x], dim=1)
            
            # Update mask
            if mask is not None:
                periodogram_mask = torch.ones(
                    batch_size, self.num_periodogram * self.num_periods_periodogram, 1,
                    device=device, dtype=dtype
                )
                mask = torch.cat([periodogram_mask, mask], dim=1)
                
        return x, mask
    
    def _add_token(self, x, mask, band=0):
        """Add learnable band-specific token to the sequence."""
        batch_size, device, dtype = x.shape[0], x.device, x.dtype
        
        # Generate token for specific band
        token = self.token_lc[band](batch_size).to(device=device, dtype=dtype)
        x = torch.cat([token, x], dim=1)
        
        # Update mask
        if mask is not None:
            token_mask = torch.ones(batch_size, 1, 1, device=device, dtype=dtype)
            mask = torch.cat([token_mask, mask], dim=1)
            
        return x, mask
    
    def _add_all_features(self, x, mask, period=None, metadata=None, 
                         features=None, periodogram=None):
        """Add all optional features to the sequence in order."""
        # Add features in specific order
        x, mask = self._add_metadata(x, mask, metadata)
        x, mask = self._add_features(x, mask, features)
        x, mask = self._add_period(x, mask, period)
        x, mask = self._add_periodogram(x, mask, periodogram)
        
        return x, mask
    
    def _process_single_lightcurve(self, x_mod, m_mod, period=None, metadata=None, 
                                  features=None, periodogram=None):
        """Process a single light curve with all features."""
        # Add tabular features
        x_enhanced, m_enhanced = self._add_all_features(
            x_mod, m_mod, period, metadata, features, periodogram
        )
        
        # Add token
        x_final, m_final = self._add_token(x_enhanced, m_enhanced, band=0)
        
        return x_final, m_final


    def forward(self, data, time, mask=None, bands=None, period=None, 
               metadata=None, features=None, periodogram=None, **kwargs):
        """
        Forward pass through the EFMHSA network.
        
        Args:
            data: Light curve flux values [batch, seq_len]
            time: Time stamps [batch, seq_len, 1]
            mask: Attention mask [batch, seq_len, 1]
            bands: Band information [batch, seq_len]
            period: Period information [batch]
            metadata: Metadata features [batch, num_metadata]
            features: Engineered features [batch, num_features]
            periodogram: Periodogram data [batch, num_periodogram]
            
        Returns:
            token: Output token representation [batch, embedding_size]
        """
        # Encode temporal information
        x_mod, m_mod, _ = self.time_encoder(
            x=data, t=time, mask=mask, band_info=bands
        )
        
        # Process light curve with all features
        x_processed, m_processed = self._process_single_lightcurve(
            x_mod, m_mod, period, metadata, features, periodogram
        )
        
        # Apply random tabular masking if enabled
        if self.mask_tabular_during_training:
            x_processed, m_processed = self._apply_random_tabular_masking(
                x_processed, m_processed, metadata, features, period, periodogram
            )
        
        # Apply transformer
        token = self.transformer_lc(x=x_processed, mask=m_processed)
        
        # Apply k-sparse attention if configured
        if self.use_k_sparse > 0:
            token = self.k_sparse(token)
            
        return token
    def get_attn_scores(self, data,time,mask,bands,metadata,features):
        """
        Get attention scores from the transformer.
        
        Args:
            data: Light curve flux values [batch, seq_len]
            time: Time stamps [batch, seq_len, 1]
            mask: Attention mask [batch, seq_len, 1]
            bands: Band information [batch, seq_len]
            metadata: Metadata features [batch, num_metadata]
            features: Engineered features [batch, num_features]
            
        Returns:
            attn_scores: Attention scores from the transformer
        """
        x_mod, m_mod, _ = self.time_encoder(
            x=data, t=time, mask=mask, band_info=bands
        )
        
        x_processed, m_processed = self._process_single_lightcurve(
            x_mod, m_mod, metadata=metadata, features=features
        )
        
        _,attn_scores = self.transformer_lc(
            x=x_processed, mask=m_processed, get_attn=True
        )
        return attn_scores
    def get_attn_scores_feat(self, data, time, mask, bands, metadata=None, features=None):
        """Get attention scores for tabular features only."""
        all_attns = []
        attns = self.get_attn_scores(data, time, mask, bands, metadata, features)
        # Calculate the end index for tabular features
        tabular_end = 1+ self.num_metadata + self.num_features
        
        for attn in attns:
            att_proc = attn[:,:,:,:tabular_end]
            all_attns.append(att_proc)
        
        return all_attns
    
    def get_attn_scores_lc(self, data, time, mask, bands, metadata=None, features=None):
        """Get attention scores for light curve data only."""
        all_attns = []
        attns = self.get_attn_scores(data, time, mask, bands, metadata, features)
        


        return attns