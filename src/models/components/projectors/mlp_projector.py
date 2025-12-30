import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, embedding_size, projection_size, **kwargs):
        super(MLPProjector, self).__init__()
        dimension = embedding_size
        self.classifier_chunk = nn.Sequential(
            nn.Linear(in_features=int(dimension), out_features=int(projection_size)),
            nn.LayerNorm(int(projection_size)),
            nn.ReLU(True),
            nn.Linear(in_features=int(projection_size), out_features=int(projection_size)),
            nn.BatchNorm1d(int(projection_size)),
            nn.ReLU(True),
            nn.Linear(in_features=int(projection_size), out_features=int(projection_size), bias=False),
        )
        self.init_model()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, x):
        return self.classifier_chunk(x)
