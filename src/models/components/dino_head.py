import torch.nn as nn

class DINOHead(nn.Module):
    def __init__(self, embedding_size, projection_size, use_bn=False,
                norm_last_layer=False, nlayers=3, hidden_size=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(embedding_size, bottleneck_dim)
        else:
            layers = [nn.Linear(embedding_size, hidden_size)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_size, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self._init_weights
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, projection_size, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.projection_size = projection_size
    def _init_weights(self, m):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0, 0.02)
            else:
                nn.init.constant_(p, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x