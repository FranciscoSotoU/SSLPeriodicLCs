"""
Collate functions for processing batches of data.

This module contains functions for collating data samples from datasets into batches,
including specialized functions for time series data (light curves).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Any, Tuple


def collate_trim_to_max_len(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that trims time series data to the maximum length in the batch.
    
    This function is designed for astronomical time series data (light curves) where each 
    sample may have a different number of observations. It finds the actual maximum length
    of data in the batch (ignoring padding) and trims all sequences to that length.
    
    Args:
        batch (List[Dict[str, torch.Tensor]]): List of dictionaries, each containing the 
                                              data for one sample
    
    Returns:
        Dict[str, torch.Tensor]: Collated batch with all time series trimmed to the same length
    """
    # Identify time series keys that need trimming
    ts_keys = ['data', 'time', 'mask', 'bands']
    
    # Find maximum actual length in the batch (ignoring padded zeros)
    max_actual_len = 0
    for sample in batch:
        if 'mask' in sample:
            # Find the actual data length (where mask is non-zero)
            actual_len = torch.sum(sample['mask'] > 0).item()
            max_actual_len = max(max_actual_len, actual_len)
    
    # If no time series data was found, default to standard collation
    if max_actual_len == 0:
        return default_collate_dict(batch)
    
    # Trim and collate the batch
    trimmed_batch = []
    for sample in batch:
        trimmed_sample = {k: v for k, v in sample.items()}
        
        # Trim time series data if present
        for key in ts_keys:
            if key in sample:
                # Keep only the first max_actual_len elements
                if len(sample[key]) > max_actual_len:
                    trimmed_sample[key] = sample[key][:max_actual_len]
                # If the length is less than max_actual_len, pad up to max_actual_len
                #elif len(sample[key]) < max_actual_len:
                #    pad_value = 0
                #    pad_size = max_actual_len - len(sample[key])
                #    padding = torch.zeros(pad_size, dtype=sample[key].dtype, device=sample[key].device)
                #    trimmed_sample[key] = torch.cat([sample[key], padding])
        
        trimmed_batch.append(trimmed_sample)

    return default_collate_dict(trimmed_batch)


def default_collate_dict(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Default collation function for dictionaries of tensors.
    
    Args:
        batch (List[Dict[str, Any]]): List of dictionaries, each containing 
                                     the data for one sample
    
    Returns:
        Dict[str, torch.Tensor]: Collated batch where each value is a batched tensor
    """
    elem = batch[0]
    collated_batch = {}
    
    for key in elem:
        if key == 'idx':
            # Special handling for indices - we keep them as a list
            collated_batch[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            # Stack tensors into a batch
            try:
                collated_batch[key] = torch.stack([d[key] for d in batch])
            except RuntimeError:
                # Handle tensors with different shapes (usually happens with time series data)
                # This is a fallback but should not happen with proper pre-trimming
                max_len = max(d[key].shape[0] for d in batch)
                padded_items = []
                
                for d in batch:
                    tensor = d[key]
                    if tensor.shape[0] < max_len:
                        # Pad to max length
                        padding_size = (0, 0) * (len(tensor.shape) - 1) + (0, max_len - tensor.shape[0])
                        padded_tensor = F.pad(tensor, padding_size, "constant", 0)
                        padded_items.append(padded_tensor)
                    else:
                        padded_items.append(tensor)
                
                collated_batch[key] = torch.stack(padded_items)
        elif isinstance(elem[key], (int, float, str, bool)):
            # Collect scalars into a list or tensor
            if isinstance(elem[key], (int, float)):
                collated_batch[key] = torch.tensor([d[key] for d in batch])
            else:
                collated_batch[key] = [d[key] for d in batch]
        elif elem[key] is None:
            # Handle None values
            collated_batch[key] = None
        else:
            # Fallback for other types
            collated_batch[key] = [d[key] for d in batch]
    
    return collated_batch


def collate_lite(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simplified collation function that just stacks tensors without padding.
    
    Use this for data that has been pre-processed to have uniform lengths.
    
    Args:
        batch (List[Dict[str, Any]]): List of dictionaries, each containing 
                                     the data for one sample
    
    Returns:
        Dict[str, Any]: Collated batch with stacked tensors
    """
    elem = batch[0]
    collated_batch = {}
    
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated_batch[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], (int, float)):
            collated_batch[key] = torch.tensor([d[key] for d in batch])
        else:
            collated_batch[key] = [d[key] for d in batch]
    
    return collated_batch


def collate_dual_dict_trim(batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Collate function that handles tuples of two data dictionaries and trims time series 
    data to the maximum length found within each dictionary type separately.
    
    This function is designed for self-supervised learning where each sample returns
    two versions of the same data (e.g., different augmentations). Each dictionary
    type is trimmed to its own optimal length for flexible processing.
    
    Args:
        batch (List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]): 
            List of tuples, each containing two dictionaries with data for one sample
    
    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: 
            Two collated batches with time series trimmed to their respective optimal lengths
    """
    # Identify time series keys that need trimming
    ts_keys = ['data', 'time', 'mask', 'bands']
    
    # Find maximum actual length for each dictionary type separately
    max_actual_len_1 = 0
    max_actual_len_2 = 0
    
    for dict1, dict2 in batch:
        # Check first dictionary
        if 'mask' in dict1:
            actual_len_1 = dict1['mask'].shape[0]
            max_actual_len_1 = max(max_actual_len_1, actual_len_1)
        
        # Check second dictionary
        if 'mask' in dict2:
            actual_len_2 = dict2['mask'].shape[0]
            max_actual_len_2 = max(max_actual_len_2, actual_len_2)
    
    # If no time series data was found, use default collation
    if max_actual_len_1 == 0 and max_actual_len_2 == 0:
        batch1 = [item[0] for item in batch]
        batch2 = [item[1] for item in batch]
        return default_collate_dict(batch1), default_collate_dict(batch2)
    
    # Trim dictionaries to their respective optimal lengths
    trimmed_batch1 = []
    trimmed_batch2 = []
    
    for dict1, dict2 in batch:
        # Process first dictionary with its optimal length
        trimmed_dict1 = {k: v for k, v in dict1.items()}
        if max_actual_len_1 > 0:
            for key in ts_keys:
                if key in dict1:
                    if len(dict1[key]) > max_actual_len_1:
                        trimmed_dict1[key] = dict1[key][:max_actual_len_1]
        
        trimmed_batch1.append(trimmed_dict1)
        
        # Process second dictionary with its optimal length
        trimmed_dict2 = {k: v for k, v in dict2.items()}
        if max_actual_len_2 > 0:
            for key in ts_keys:
                if key in dict2:
                    if len(dict2[key]) > max_actual_len_2:
                        trimmed_dict2[key] = dict2[key][:max_actual_len_2]
        trimmed_batch2.append(trimmed_dict2)
    
    # Collate both trimmed batches
    collated_batch1 = default_collate_dict(trimmed_batch1)
    collated_batch2 = default_collate_dict(trimmed_batch2)
    
    return collated_batch1, collated_batch2

def collate_patch_sequences(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for patch sequences in time series data.
    
    This function is designed to handle batches of light curve data that have been
    processed into patches. It collates the patches into a single batch while ensuring
    that the sequence lengths are consistent across the batch.
    
    Args:
        batch (List[Dict[str, torch.Tensor]]): List of dictionaries, each containing 
                                              the data for one sample
    
    Returns:
        Dict[str, torch.Tensor]: Collated batch with patches processed and stacked
    """
    # Use default collation for simplicity
    return default_collate_dict(batch)


def collate_dino(batch: List[List[Dict[str, torch.Tensor]]]) -> List[Dict[str, torch.Tensor]]:
    """
    Collate function for DINO training that handles multiple views per sample.
    
    This function is used when training with the DINO self-supervised learning method,
    where each sample contains multiple views (global and local crops).
    
    Args:
        batch (List[List[Dict[str, torch.Tensor]]]): List of samples, where each sample
                                                    is a list of view dictionaries
    
    Returns:
        List[Dict[str, torch.Tensor]]: List of collated batches, one for each view
    """
    # Each item in batch is a list of views for one sample
    # We need to reorganize to have one batch per view
    
    if not batch:
        return []
    
    num_views = len(batch[0])  # Number of views per sample
    view_batches = []
    
    for view_idx in range(num_views):
        # Collect all samples for this view
        view_batch = [sample[view_idx] for sample in batch]
        # Collate this view's batch
        collated_view = default_collate_dict(view_batch)
        view_batches.append(collated_view)
    
    return view_batches