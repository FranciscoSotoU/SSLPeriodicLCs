import torch
import torch.nn as nn
import numpy as np

class Harmonics(nn.Module):
    """Module that generates harmonic sequences based on specified patterns."""
    
    def __init__(self, num_periods=4,**kwargs):
        super(Harmonics, self).__init__()
        # Generate harmonics once during initialization and register as parameter
        self.values = nn.Parameter(
            harmonics(num_periods,**kwargs), requires_grad=False
        )

    def forward(self):
        return self.values


def harmonics(num_periods, p_mode="sub_harmonics", **kwargs):
    """
    Generate various harmonic patterns based on the specified mode.
    
    Args:
        num_periods: Number of periods to generate
        p_mode: Harmonic pattern mode ('one', 'two', 'geometric', 'sub_harmonics', 'mix_harmonics')
        
    Returns:
        torch.Tensor: Tensor of harmonic values
    """
    # Remove unnecessary print statement
    if num_periods <=1:
        return torch.ones(1, dtype=torch.float32)
    if p_mode == "one":
        # Vectorized creation - all ones
        return torch.ones(num_periods, dtype=torch.float32)
        
    elif p_mode == "two":
        # More efficient implementation using torch directly
        result = torch.ones(num_periods, dtype=torch.float32)
        half_size = num_periods // 2
        result[half_size:] = 2.0
        return result
        
    elif p_mode == "geometric":
        # Vectorized implementation
        result = torch.ones(num_periods, dtype=torch.float32)
        
        if num_periods >= 2:
            result[1] = 2.0
            
            if num_periods > 2:
                # For indices 2 and above, apply 4^i
                indices = torch.arange(2, num_periods, dtype=torch.float32)
                result[2:] = 4.0 ** indices
        
        # Sort the result
        return result.sort()[0]
        
    elif p_mode == "sub_harmonics":
        # Calculate first half: 1/2^i for i=0,1,...,num_periods//2-1
        half_size = num_periods // 2
        powers_neg = torch.arange(0, half_size, dtype=torch.float32)
        first_half = 1.0 / (2.0 ** powers_neg)
        
        # Calculate second half: 2^i for i=0,1,...,num_periods//2-1
        powers_pos = torch.arange(0, half_size, dtype=torch.float32)
        second_half = 2.0 ** powers_pos
        
        # Concatenate and sort
        result = torch.cat([first_half, second_half])
        return result.sort()[0]
        
    elif p_mode == "mix_harmonics":
        # More efficient implementation
        result = torch.zeros(num_periods, dtype=torch.float32)
        half_size = num_periods // 2
        
        # Set first half to powers of 2
        indices = torch.arange(half_size, dtype=torch.float32)
        result[:half_size] = 2.0 ** indices
        
        # Sort the result
        return result.sort()[0]
    
    # Default case - return ones
    return torch.ones(num_periods, dtype=torch.float32)