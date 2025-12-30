import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KSparse(nn.Module):
    """
    A module that enforces sparsity on its input.
    Only the top percentage of largest values in each row are kept, others are set to zero.
    Designed for MLP-style input (batch_size, features).
    """
    def __init__(self, k_sparse_percentage: float, **kwargs):
        """
        Args:
            k_sparse_percentage (float): Percentage of non-zero elements to keep in each row (0.0 to 1.0)
        """
        super().__init__()
        if not 0 <= k_sparse_percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        self.percentage = k_sparse_percentage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that enforces sparsity based on percentage.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features)
            mask (torch.Tensor): Mask tensor of shape (batch_size, features)

        Returns:
            torch.Tensor: Sparse version of the input tensor
        """
        if self.percentage <= 0:
            return x
        # Calculate number of elements to keep based on percentage
        num_features = x.size(1)
        k = max(1, math.ceil(num_features * self.percentage))
        # Get value at the k-th position for each row
        kth_values, _ = torch.kthvalue(x, num_features - k + 1, dim=1, keepdim=True)
        
        # Create mask for values larger than k-th value
        mask = (x >= kth_values).float()
        # Apply mask to input
        return x * mask

