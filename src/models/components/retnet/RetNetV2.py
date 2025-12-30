from fla import MultiScaleRetention
import torch.nn as nn
from fla.ops.utils.index import prepare_cu_seqlens_from_mask

class RetNet(nn.Module):
    def __init__(
            self, embedding_size=272, num_encoders=2, expansion_coef=4, num_heads=16, norm_type="layernorm", double_v_dim=False, scale_time=180, **kwargs
        ):
        print(f"RetNet: embedding_size={embedding_size}, num_encoders={num_encoders}, expansion_coef={expansion_coef}, num_heads={num_heads}, norm_type={norm_type}, double_v_dim={double_v_dim}, scale_time={scale_time}")
        super(RetNet, self).__init__()
        self.layers = num_encoders
        hidden_dim = embedding_size
        self.hidden_dim = hidden_dim
        self.expansion_coef = expansion_coef
        self.ffn_size = hidden_dim * expansion_coef
        self.heads = num_heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim
        self.retentions = nn.ModuleList([
            MultiScaleRetention(mode='fused_chunk',hidden_size=hidden_dim, num_heads=self.heads, layer_idx=i, fuse_norm=True)
            for i in range(self.layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, self.ffn_size),
                nn.GELU(),
                nn.Linear(self.ffn_size, hidden_dim)
            )
            for i in range(self.layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for i in range(self.layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for i  in range(self.layers)
        ])
        self.output = hidden_dim

    def forward(self, X, attention_mask=None):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        # For batched input, we should not use cu_seqlens
        for i in range(self.layers):
            X = self.layer_norms_1[i](X)
            batch, seq_len, emb_dim = X.shape
            X = X.view(1, -1, self.hidden_dim)  # Reshape to (1, batch_size * seq_len, hidden_dim)
            X_out = self.retentions[i](hidden_states=X, attention_mask=attention_mask)[0]
            X = X_out.view(batch, seq_len, emb_dim)
            X = self.layer_norms_2[i](X)
            X = self.ffns[i](X) + X
        return X
