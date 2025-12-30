import torch
import torch.nn as nn


from .transformer2 import Transformer, TimeHandlerParallel, Harmonics, Token, Embedding, TimeHandler
from .utils import KSparse
import copy



class ATAT(nn.Module):
    def __init__(self, **kwargs):
        super(ATAT, self).__init__()

        # Time encoders
        
        self.use_metadata = kwargs.get("use_metadata", False)
        self.use_features = kwargs.get("use_features", False)
        self.use_period = kwargs.get("use_period", False)
        self.use_lightcurve = kwargs.get("use_lightcurve", True)
        self.embedding_size_feat = kwargs.get("embedding_size_feat", 128)
        self.embedding_size_lc = kwargs.get("embedding_size_lc", 192)
        if self.use_lightcurve:
            kwargs_lc = copy.deepcopy(kwargs)
            kwargs_lc["embedding_size"] = self.embedding_size_lc
            kwargs_lc["num_encoders"] = kwargs.get("num_encoders", 3)
            
            self.transformer_lc = Transformer(**kwargs_lc)
            self.token_lc = Token(**kwargs_lc)
            self.time_encoder = TimeHandler(**kwargs_lc)
            self.band_proc = kwargs.get("band_proc", None)
            self.num_bands = kwargs.get("num_bands", 2)

        if self.use_metadata or self.use_features or self.use_period:
            num_metadata = kwargs.get("num_metadata") if self.use_metadata else 0
            num_features = kwargs.get("num_features") if self.use_features else 0
            num_periods = kwargs.get("num_periods") if self.use_period else 0
            kwargs_feat = copy.deepcopy(kwargs)
            kwargs_feat["embedding_size"] = self.embedding_size_feat
            kwargs_feat["num_encoders"] = kwargs.get("num_encoders", 3)
            kwargs_feat['length_size'] = num_metadata + num_features + num_periods
            self.transformer_feat = Transformer(**kwargs_feat)
            self.token_feat = Token(**kwargs_feat)
            self.embedding_feat = Embedding(**kwargs_feat)
        if self.use_period:
            self.harmonics = Harmonics(**kwargs_feat)

        self.init_model()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p,0,0.02)

    def add_token(self, x, mask=None, token=None):
        batch_size, seq_len, dim = x.size()
        x = torch.cat([token,x], dim=1)
        m_token =torch.ones(batch_size, 1,1).to(x.device)
        m = torch.cat([m_token, mask], dim=1)
        return x, m

    def _process_lc(self, x_mod, m_mod):
        batch_size,_,_ = x_mod.size()
        x_ , m_ = self.add_token(x=x_mod, mask=m_mod, token=self.token_lc(batch_size))
        return x_, m_

    def _process_metadata(self, metadata=None, features=None,period=None):
        tabular_ = []
        if metadata is not None and self.use_metadata:
            tabular_.append(metadata)
        if features is not None and self.use_features:
            tabular_.append(features)
        if period is not None and self.use_period:
            harmonics = self.harmonics()
            period_expanded = period.view(-1, 1)
            harmonic_periods = (harmonics * period_expanded)
            tabular_.append(harmonic_periods)
        if len(tabular_) > 0:
            tabular_feat = torch.cat(tabular_, dim=1).unsqueeze(-1)
        else:
            tabular_feat = tabular_[0].unsqueeze(-1)
        batch_size, _, _ = tabular_feat.size()
        token = self.token_feat(batch_size)
        tabular_mask = torch.ones_like(tabular_feat)
        tabular_feat = self.embedding_feat(tabular_feat)
        tabular_feat = torch.cat([token, tabular_feat], dim=1)
        token_mask = torch.ones(batch_size, 1,1).to(tabular_feat.device)
        tabular_mask = torch.cat([token_mask, tabular_mask], dim=1)
        return tabular_feat, tabular_mask



    def forward(self, data, time, mask=None, metadata=None, features=None, bands=None,period=None, **kwargs):
        tokens = []
        if self.use_metadata or self.use_features or self.use_period:
            x_mod_feat, m_mod_feat = self._process_metadata(metadata=metadata, features=features, period=period)
            token_feat = self.transformer_feat(
                x=x_mod_feat,
                mask=m_mod_feat
            )
            tokens.append(token_feat)
        if self.use_lightcurve:
            
            x_mod, m_mod, _ = self.time_encoder(x=data, t=time, mask=mask, band_info=bands)

            x_mod_token, m_mod_token = self._process_lc(
                x_mod, m_mod
            )
        
            token_lc = self.transformer_lc(
                x=x_mod_token,
                mask=m_mod_token
            )
            tokens.append(token_lc)
        if len(tokens) > 1:
            tokens = torch.cat(tokens, dim=1)
        else:
            tokens = tokens[0]
        return tokens
    def get_attn_scores_feat(self, data=None,time = None,mask=None,bands=None,metadata=None, features=None):

        x_mod_feat, m_mod_feat = self._process_metadata(metadata=metadata, features=features)
        _, attn_ft = self.transformer_feat(
            x=x_mod_feat,
            mask=m_mod_feat,
            get_attn=True
        )
        return attn_ft


            
    def get_attn_scores_lc(self, data=None, time=None, mask=None, bands=None, metadata=None, features=None):
        x_mod, m_mod, _ = self.time_encoder(x=data, t=time, mask=mask, band_info=bands)
        
        x_mod_token, m_mod_token = self._process_lc(
            x_mod, m_mod
        )

        _, attn_lc = self.transformer_lc(
            x=x_mod_token,
            mask=m_mod_token,
            get_attn=True
        )
        return attn_lc
    




