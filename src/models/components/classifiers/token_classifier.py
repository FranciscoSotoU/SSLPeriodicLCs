import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes,num_encoders, **kwargs):
        super(TokenClassifier, self).__init__()
        emb_size_feat = kwargs.get("embedding_size_feat", 0) if (kwargs.get("use_features", False) or kwargs.get("use_metadata", False)) else 0
        emb_size_lc = kwargs.get("embedding_size_lc", 0) if kwargs.get("use_lightcurve", True) else 0
        emb_size = emb_size_feat + emb_size_lc
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(in_features=int(emb_size), out_features=num_classes)
    
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)
