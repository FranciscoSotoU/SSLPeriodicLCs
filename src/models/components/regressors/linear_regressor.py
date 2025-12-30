import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressor(nn.Module):
    def __init__(self,  embedding_size, num_classes, **kwargs):
        super(LinearRegressor, self).__init__()

        self.layer = nn.Linear(embedding_size,num_classes)
        self.norm = nn.LayerNorm(embedding_size)
        self.init_model()
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)

    def forward(self, x):
        return self.layer(self.norm(x))
