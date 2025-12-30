import torch
import torch.nn as nn

class Token(nn.Module):
    def __init__(self, **kwargs):
        super(Token, self).__init__()
        
        embedding_size = kwargs.get("embedding_size", 512)
        self.token = nn.parameter.Parameter(
            torch.randn(1, 1, embedding_size), requires_grad=True
        )

    def forward(self, batch_size):
        return self.token.repeat(batch_size, 1, 1)
