import copy
import torch
import torch.nn as nn


from .transformer import Transformer, TimeHandlerParallel, Harmonics, Token, Embedding

class MHSPANR(nn.Module):
    def __init__(self, **kwargs):
        super(MHSPANR, self).__init__()

        # Time encoders
        self.time_encoder = TimeHandlerParallel(**kwargs)
        # Transformers
        self.transformer_lc = Transformer(**kwargs)
        self.transformer_feat  = Transformer(**kwargs)
        # Tokens
        token = Token(**kwargs)
        self.token_lc = [copy.deepcopy(token) for _ in range(2)]
        self.token_lc = nn.ModuleList(self.token_lc)
        self.token_feat = Token(**kwargs)
        self.feat_encoder = Embedding(**kwargs)
        self.harmonics = Harmonics(**kwargs)
        self.num_features = kwargs.get("num_features", 0)
        self.num_metadata = kwargs.get("num_metadata", 0)
        self.init_model()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _process_bands(self, x_mod, m_mod, t_mod):
        """Core processing logic shared between forward and get_tokens"""
        # Compute all colors at once
        x_mod_token = []
        m_mod_token = []
        t_mod_token = []
        
        # Process each band in one loop to avoid redundancy
        for i, (x, m, t) in enumerate(zip(x_mod, m_mod, t_mod)):
            x_, m_, t_ = self.add_token(x=x , mask=m, band=i, t=t)
            
            x_mod_token.append(x_)
            m_mod_token.append(m_)
            t_mod_token.append(t_)
        
        return x_mod_token, m_mod_token, t_mod_token

    def add_token(self, x, mask=None, t=None, band=0, **kwargs):
        batch_size = x.shape[0]
        device = x.device
        
        # Generate token once
        token = self.token_lc[band](batch_size).to(device=device)
        x = torch.cat([token, x], axis=1)

        # Create tokens for t and mask in one go if needed
        if t is not None:
            t_token = torch.zeros(batch_size, 1, 1, device=device)
            t = torch.cat([t_token, t], axis=1)
            
        if mask is not None:
            m_token = torch.ones(batch_size, 1, 1, device=device)
            mask = torch.cat([m_token, mask], axis=1)

        return x, mask, t
    def add_token_feat(self, x, mask=None):
        batch_size = x.shape[0]
        device = x.device
        
        # Generate token once
        token = self.token_feat(batch_size).to(device=device)

        x = torch.cat([token, x], axis=1)

        # Create tokens for t and mask in one go if needed
        if mask is not None:
            m_token = torch.ones(batch_size, 1, 1, device=device)
            mask = torch.cat([m_token, mask], axis=1)


        return x, mask
    

    def forward(self, data, time, mask=None, period=None, metadata=None, features=None, **kwargs):
        # Light curve Modulation
        x_mod, m_mod, t_mod = self.time_encoder(x=data, t=time, mask=mask)
        
        # Process all bands
        x_mod_token, m_mod_token, t_mod_token = self._process_bands(
            x_mod, m_mod, t_mod)
        
        # Transform each band and concatenate results
        tokens = []
        for i in range(len(x_mod_token)):
            tokens.append(
                    self.transformer_lc(
                        query=x_mod_token[i],
                        key=x_mod_token[i],
                        value=x_mod_token[i],
                        t=t_mod_token[i],
                        mask=m_mod_token[i],
                        p=self.harmonics() * period.view(-1,1),),
            )

        tokens = torch.cat(tokens, axis=1)
        #add a new dimension to metadata to be batch,seq_len,1
        tab_data = []
        if self.num_metadata > 0:
            metadata = metadata.unsqueeze(1)
            tab_data.append(metadata)
        if self.num_features > 0:
            features = features.unsqueeze(1)
            tab_data.append(features)
        if self.num_metadata > 0 or self.num_features > 0:
            metadata = torch.cat(tab_data, axis=1)
            feat_token,_ = self.add_token_feat(metadata)
            feat_tokens = self.transformer_feat(
                query=feat_token,
                key=feat_token,
                value=feat_token,
                mask=None,
            )
        tokens = torch.cat([tokens, feat_tokens], axis=1)

        return tokens