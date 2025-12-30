import torch 
import torch.nn as nn
import torch.nn.functional as F


class AttentionWeighting(torch.nn.Module):
    """Attention mechanism to weight different modalities."""
    
    def __init__(self, feature_dim, num_heads=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Attention projection layers
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, modalities):
        """
        Args:
            modalities: List of tensors, each of shape (batch, feature_dim)
        
        Returns:
            Weighted combination of modalities
        """
        # Stack modalities: (batch, num_modalities, feature_dim)
        x = torch.stack(modalities, dim=1)
        
        batch_size, num_modalities, feature_dim = x.shape
        
        # Compute attention scores
        q = self.query(x)  # (batch, num_modalities, feature_dim)
        k = self.key(x)    # (batch, num_modalities, feature_dim)
        v = self.value(x)  # (batch, num_modalities, feature_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights
        weighted = torch.matmul(attention_weights, v)
        
        # Average across modalities
        output = weighted.mean(dim=1)
        
        return output


class KSparsityTails(torch.nn.Module):

    def __init__(self, p=0.25):
        super().__init__()
        self.p = p
    def forward(self, z):
        k = int(self.p * z.shape[-1])
        _, post_indices = torch.topk(z, k, dim=-1)
        _, neg_indices = torch.topk(-z, k, dim=-1)

        indices = torch.hstack([post_indices, neg_indices])
        sparsity_mask = torch.zeros_like(z)
        sparsity_mask.scatter_(1, indices, 1)
        return z * sparsity_mask


class KSparsity(torch.nn.Module):

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, z):
        
        k = int(self.p * z.shape[-1])
        _, indices = torch.topk(z, k, dim=-1)
        sparsity_mask = torch.zeros_like(z)
        sparsity_mask.scatter_(1, indices, 1)

        return z * sparsity_mask


class Concatenation(object):

    def __init__(self, encode_function=torch.nn.Identity(), partial=True):
        self.encode_function = encode_function
        self.partial = partial
    
    def validate_modaity(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        if len(x.shape) == 2:
            assert ValueError(
                    "2D batchxfeature is not weel formated"
                )
    
    def __call__(self, sources):
        """_summary_

        Args:
            sources (_type_): a list with 2D tensor rank

        Returns:
            _type_: _description_
        """
        modalities = []

        if self.partial:
            
            for z in sources:
                self.validate_modaity(z)
                modalities.append(self.encode_function(z))
            
            return torch.concat(modalities, dim=-1)
        
        for z in sources:
            self.validate_modaity(z)
            modalities.append(z)

        return self.encode_function(torch.concat(modalities, dim=-1))


class ConcatenationRelu(Concatenation):
    def __init__(self, encode_function=torch.nn.ReLU()):
        super().__init__(encode_function=encode_function)
        print("ConcatenationRelu instantiated")
    
class ConcatenationDropOut(Concatenation):
    def __init__(self, encode_function=torch.nn.Dropout1d(p=0.1)):
        super().__init__(encode_function=encode_function)
        print("ConcatenationDropOut instantiated")
    

class ConcatenationKSparse(Concatenation):
    def __init__(self, encode_function=KSparsity(p=0.5)):
        super().__init__(encode_function=encode_function)
        print("ConcatenationKSparse instantiated")
    

class ConcatenationKSparseTwoTails(Concatenation):
    def __init__(self, encode_function=KSparsityTails(p=0.25)):
        super().__init__(encode_function=encode_function)
        print("ConcatenationKSparseTwoTails instantiated")


class ConcatenationNormalization(torch.nn.Module):
    """Concatenation with layer normalization as an nn.Module to ensure proper device handling."""
    
    def __init__(self, normalized_shape=128, partial=True, n=2):
        super().__init__()
        # Use ModuleList to properly register submodules
        self.encode_function_list = nn.ModuleList([
            nn.LayerNorm(normalized_shape=normalized_shape) for _ in range(n)
        ])
        self.partial = partial
        print("ConcatenationNormalization instantiated")
    
    def validate_modaity(self, x):
        """Validate modality tensor shape."""
        if len(x.shape) == 2:
            assert ValueError("2D batchxfeature is not well formatted")
    
    def forward(self, sources):
        """Forward pass for concatenation with normalization.
        
        Args:
            sources: List of tensors to concatenate
            
        Returns:
            Concatenated and optionally normalized tensor
        """
        modalities = []
        
        if self.partial:
            for i, z in enumerate(sources):
                self.validate_modaity(z)
                modalities.append(self.encode_function_list[i](z))
            return torch.concat(modalities, dim=-1)
        
        for z in sources:
            self.validate_modaity(z)
            modalities.append(z)
        
        # For non-partial mode, concatenate first then normalize with first layer norm
        return self.encode_function_list[0](torch.concat(modalities, dim=-1))


class AttentionBasedConcatenation(torch.nn.Module):
    """Attention-based concatenation that weights modalities before combining them."""
    
    def __init__(self, feature_dim=128, num_heads=1, concat_mode='weighted'):
        """
        Args:
            feature_dim: Dimension of each modality feature
            num_heads: Number of attention heads
            concat_mode: 'weighted' (attention-weighted sum) or 'concat' (attention + concat)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.concat_mode = concat_mode
        self.attention = AttentionWeighting(feature_dim, num_heads)
        print(f"AttentionBasedConcatenation instantiated with mode: {concat_mode}")
    
    def forward(self, sources):
        """
        Args:
            sources: List of tensors, each of shape (batch, feature_dim)
        
        Returns:
            Combined representation
        """
        if self.concat_mode == 'weighted':
            # Pure attention-weighted combination
            return self.attention(sources)
        elif self.concat_mode == 'concat':
            # Attention-weighted + original concatenation
            weighted = self.attention(sources)
            concatenated = torch.concat(sources, dim=-1)
            return torch.concat([weighted, concatenated], dim=-1)
        else:
            raise ValueError(f"Unknown concat_mode: {self.concat_mode}")
    
