import torch
import torch.nn as nn


from .transformer import Transformer, TimeHandlerParallel, Harmonics, Token, Embedding
import copy



class CAMHSA(nn.Module):
    def __init__(self, **kwargs):
        super(CAMHSA, self).__init__()

        # Time encoders
        self.time_encoder = TimeHandlerParallel(**kwargs)
        self.use_period = kwargs.get("use_period", True)  # Add default value for use_period
        self.use_metadata = kwargs.get("use_metadata", True)  # Add default for use_metadata
        self.use_features = kwargs.get("use_features", False)  # Add default for use_features
        
        if self.use_period:
            self.period_encoder = Embedding(length_size=1,**kwargs)
        if self.use_metadata:
            self.num_metadata = kwargs.get("num_metadata", 0)
            self.metadata_encoder = Embedding(length_size=self.num_metadata,**kwargs)
        if self.use_features:
            self.num_features = kwargs.get("num_features", 0)
            self.feature_encoder = Embedding(length_size=self.num_features,**kwargs)

        self.transformer = Transformer(**kwargs)

        token = Token(**kwargs)
        self.token_lc = nn.ModuleList([copy.deepcopy(token) for _ in range(2)])
        self.harmonics = Harmonics(**kwargs)
        self.main = kwargs.get("main", 'lc')
        self.init_model()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def _process_sequences(self, x_mod, m_mod, period, metadata=None, features=None, main_sequence='lc'):
        batch_size = x_mod[0].shape[0]
        device = x_mod[0].device
        
        # Process light curves
        lc_sequences = []
        lc_masks = []
        for i, (x, m) in enumerate(zip(x_mod, m_mod)):
            # Add token to light curve sequence
            x_, m_ = self.add_token(x=x, mask=m, band=i)
            lc_sequences.append(x_)
            lc_masks.append(m_)
        
        # Process period information
        # Initialize meta sequence and mask as None
        meta_sequence = None
        meta_mask = None
        
        # Process period information if enabled
        if self.use_period and period is not None:
            period = period.to(x_mod[0].dtype)
            harmonics = self.harmonics()
            period_expanded = period.view(-1, 1)
            harmonic_periods = harmonics * period_expanded
            harmonic_periods = harmonic_periods.unsqueeze(-1)
            p_enc = self.period_encoder(harmonic_periods)
            p_mask = torch.ones(batch_size, harmonic_periods.shape[1], 1, device=device)
            
            # Initialize meta sequence with period encoding
            meta_sequence = p_enc
            meta_mask = p_mask

        # Process metadata if available and enabled
        if self.use_metadata and metadata is not None:
            metadata = metadata.unsqueeze(1).view(batch_size, -1, 1)
            metadata_enc = self.metadata_encoder(metadata)
            m_metadata = torch.ones(batch_size, metadata.shape[1], 1, device=device)
            
            # Append or initialize meta sequence
            if meta_sequence is not None:
                meta_sequence = torch.cat([meta_sequence, metadata_enc], dim=1)
                meta_mask = torch.cat([meta_mask, m_metadata], dim=1)
            else:
                meta_sequence = metadata_enc
                meta_mask = m_metadata

        # Process features if available and enabled
        if self.use_features and features is not None:
            features = features.unsqueeze(1).view(batch_size, -1, 1)
            features_enc = self.feature_encoder(features)
            m_features = torch.ones(batch_size, features.shape[1], 1, device=device)
            
            # Append or initialize meta sequence
            if meta_sequence is not None:
                meta_sequence = torch.cat([meta_sequence, features_enc], dim=1)
                meta_mask = torch.cat([meta_mask, m_features], dim=1)
            else:
                meta_sequence = features_enc
                meta_mask = m_features
        # Add token to metadata sequence if needed based on main_sequence type
        if main_sequence in ['meta', 'both', 'all'] and meta_sequence is not None:
            meta_sequence, meta_mask = self.add_token(x=meta_sequence, mask=meta_mask)
        
        return {
            'lc_sequences': lc_sequences,
            'lc_masks': lc_masks,
            'meta_sequence': meta_sequence,
            'meta_mask': meta_mask,
            'main_sequence_type': main_sequence
        }

    def add_token(self, x, mask=None, band=0):
        batch_size = x.shape[0]
        device = x.device
        
        # Generate token once
        token = self.token_lc[band](batch_size).to(device=device)
        x = torch.cat([token, x], axis=1)

        if mask is not None:
            m_token = torch.ones(batch_size, 1, 1, device=device)
            mask = torch.cat([m_token, mask], axis=1)

        return x, mask
    

    def forward(self, data, time, mask=None, period=None, metadata=None, features=None, **kwargs):
        # Extract modulated data
        x_mod, m_mod, _ = self.time_encoder(x=data, t=time, mask=mask)
        n_bands = len(x_mod)
        
        # Process sequences based on the main mode
        if self.main == 'lc':
            # Light curve as main sequence
            sequences = self._process_sequences(
                x_mod, m_mod, period, metadata=metadata, features=features)
            
            # Parallel processing of all bands to improve efficiency
            tokens = []
            for i in range(n_bands):
                # For cross-attention: query from LC, key/value from meta
                token = self.transformer(
                    query=sequences['lc_sequences'][i],
                    key=sequences['meta_sequence'],
                    value=sequences['meta_sequence'],
                    mask=sequences['lc_masks'][i],
                )
                tokens.append(token)
            
            return torch.cat(tokens, dim=1)
            
        elif self.main == 'meta':
            # Metadata as main sequence
            sequences = self._process_sequences(
                x_mod, m_mod, period, metadata=metadata, features=features, main_sequence='meta')
            
            # Process each band with meta as main
            tokens = []
            for i in range(n_bands):
                # For cross-attention: query from meta, key/value from LC
                token = self.transformer(
                    query=sequences['meta_sequence'],
                    key=sequences['lc_sequences'][i],
                    value=sequences['lc_sequences'][i],
                    mask=sequences['meta_mask'],
                )
                tokens.append(token)
            
            return torch.cat(tokens, dim=1)
            
        elif self.main == 'both':
            # Both as main in different passes
            sequences = self._process_sequences(
                x_mod, m_mod, period, metadata=metadata, features=features, main_sequence='both')
            
            tokens = []
            for i in range(n_bands):
                # Meta as main, LC[i] as context
                meta_main_token = self.transformer(
                    query=sequences['meta_sequence'],
                    key=sequences['lc_sequences'][i],
                    value=sequences['lc_sequences'][i],
                    mask=sequences['meta_mask'],
                )
                tokens.append(meta_main_token)
                
                # LC[i] as main, Meta as context
                lc_main_token = self.transformer(
                    query=sequences['lc_sequences'][i],
                    key=sequences['meta_sequence'],
                    value=sequences['meta_sequence'],
                    mask=sequences['lc_masks'][i],
                )
                tokens.append(lc_main_token)
            
            return torch.cat(tokens, dim=1)
            
        elif self.main == 'all':
            # Process all combinations
            sequences = self._process_sequences(
                x_mod, m_mod, period, metadata=metadata, features=features, main_sequence='both')
            
            tokens = []
            # Batch process all transformer calls for efficiency
            for i in range(n_bands):
                # Self-attention for LC
                tokens.append(self.transformer(
                    query=sequences['lc_sequences'][i],
                    key=sequences['lc_sequences'][i],
                    value=sequences['lc_sequences'][i],
                    mask=sequences['lc_masks'][i],
                ))
                
                # Meta as main, LC as context
                tokens.append(self.transformer(
                    query=sequences['meta_sequence'],
                    key=sequences['lc_sequences'][i],
                    value=sequences['lc_sequences'][i],
                    mask=sequences['meta_mask'],
                ))
                
                # LC as main, Meta as context
                tokens.append(self.transformer(
                    query=sequences['lc_sequences'][i],
                    key=sequences['meta_sequence'],
                    value=sequences['meta_sequence'],
                    mask=sequences['lc_masks'][i],
                ))
            
            return torch.cat(tokens, dim=1)





