import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import KSparse

class MLPClassifier(nn.Module):
    def __init__(self,  embedding_size, num_classes, **kwargs):
        super(MLPClassifier, self).__init__()
        
        input_dim = embedding_size

        activation_name = kwargs.get("activation", "gelu")
        activation = nn.ReLU() if activation_name == "relu" else nn.GELU()
        
        norm_type = kwargs.get("norm", 'layer')
        norm = None
        if norm_type == "batch":
            norm = nn.BatchNorm1d(input_dim)
        elif norm_type == "layer":
            norm = nn.LayerNorm(input_dim)
        elif norm_type == 'rmsnorm':
            norm = nn.RMSNorm(input_dim)
            
        dropout_rate = kwargs.get("dropout", 0.2)
        
        # Build sequential layers
        layers = [nn.Linear(input_dim, input_dim), activation]
        if norm: layers.append(norm)
        layers.extend([nn.Dropout(dropout_rate), nn.Linear(input_dim, input_dim), activation])
        if norm: layers.append(norm)
        layers.extend([nn.Dropout(dropout_rate), nn.Linear(input_dim, num_classes)])
        
        self.classifier_chunk = nn.Sequential(*layers)

        self.init_model()
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.classifier_chunk(x)
