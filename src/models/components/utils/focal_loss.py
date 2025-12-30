import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multiclass classification.
    
    Args:
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output:
                        'none' | 'mean' | 'sum' (default: 'mean')
        ignore_index (int): Specifies a target value that is ignored (default: -100)
    """
    
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predictions (logits) of shape (N, C) where N is batch size, C is number of classes
            targets (Tensor): Ground truth labels of shape (N,) with class indices
        
        Returns:
            Tensor: Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
