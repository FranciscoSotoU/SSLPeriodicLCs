import torch
import torch.nn as nn


class LinearProjector(nn.Module):
    def __init__(self, embedding_size, projection_size, **kwargs):
        super().__init__()
        
        self.classifier_chunk = nn.Sequential(
            nn.Linear(in_features=int(embedding_size), out_features=projection_size),
        )
        #self.init_model()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.classifier_chunk(x)
  