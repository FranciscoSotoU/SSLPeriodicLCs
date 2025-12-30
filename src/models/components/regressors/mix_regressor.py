import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import KSparse

class MixRegressor(nn.Module):
    def __init__(self,  embedding_size, num_classes,dropout, **kwargs):
        super(MixRegressor, self).__init__()
        self.norm = nn.LayerNorm(embedding_size)
        self.cls = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_classes),
        )
        self.init_model()
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p,0,0.02)

    def forward(self, x):

        return self.cls(self.norm(x))
