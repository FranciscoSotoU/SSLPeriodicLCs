"""
Forced Photometry Dataset Module for Multi-Modal astronomical time series data.

This module provides a PyTorch Dataset implementation for loading and processing 
forced photometry light curve data with multiple modalities (time series data, 
metadata, features, and periodograms).
"""

import os
import logging
import yaml
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer as Scaler
from joblib import dump, load


def dataset_split_handler(set_type, split="0", reduced=False, prefix=None):
    """
    Handles dataset split naming conventions with optional prefix.
    
    Args:
        set_type (str): Dataset split type ('train', 'valid', or 'test')
        split (str): The fold number for train/valid splits
        reduced (bool): Whether reduced dataset is used (not implemented)
        prefix (str): Optional prefix for dataset name

    Returns:
        str: The formatted dataset split name
    """
    if prefix:
        if set_type == "test":
            return f"{prefix}_test"
        elif set_type == "valid":
            return f"{prefix}_valid_{split}"
        elif set_type == "train":
            return f"{prefix}_train_{split}"
        elif set_type == 'training':
            return f"{prefix}_training_{split}"
        elif set_type == 'validation':
            return f"{prefix}_validation_{split}"
    else:
        if set_type == "test":
            return "test"
        elif set_type == "valid":
            return f"validation_{split}"
        elif set_type == "train":
            return f"training_{split}"
        elif set_type == 'training':
            return f"training_{split}"
        elif set_type == 'validation':
            return f"validation_{split}"
    
class ForcedPhotometryDatasetUnsupervised(Dataset):
    """
    Multimodal Forced Photometry Dataset for Astronomical Time Series Data.

    This class implements a PyTorch Dataset for handling astronomical time series data,
    particularly light curves with various modalities like metadata, features,
    and periodograms.
    """

    def __init__(
        self,
        data_dir="",            # Directory containing the dataset files
        data_name="",           # Name of the main data file
        subset_name=None,         # Optional subset name prefix
        set_type="train",       # Dataset split type (train, valid, test)
        split="0",              # Data fold number
        use_lightcurve=True,    # Whether to include light curve data
        use_metadata=False,     # Whether to include metadata features
        use_features=False,      # Whether to include extracted features
        normalize_tab=False,    # Whether to normalize tabular data
        max_length=200,         # Maximum sequence length
        sampling_strategy='truncate',  # Sequence length strategy: 
                                      # 'truncate', 'random', 'pad_only'
        selected_features=None,  # Names of selected features
        selected_mds=None,  # Names of selected metadata columns
        norm_type=None,  # Normalization type for data
        transform_module=None,  # Optional transformations to apply
        return_snid = False,  # Whether to return SNID ( ID)
        patch_size = 0,
        dino = False,
        **kwargs,
    ):
        """
        Initialize the Forced Photometry Dataset.
        """
        # Store configuration parameters
        self.DINO = dino
        self.patch_size = patch_size
        self.max_length = max_length
        self.sampling_strategy = sampling_strategy
        self.use_lightcurve = use_lightcurve
        self.dataset_dir = data_dir
        self.normalize_tab = normalize_tab
        self.split = split
        self.set_type = set_type
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.subset_name = subset_name
        self.norm_type = norm_type
        self.transform_module = transform_module
        self.return_snid = return_snid
        # Set up data paths and load file
        data_path = os.path.join(data_dir, data_name)+".h5"
        self.data_path = data_path
        h5_ = h5py.File(data_path, "r", libver='latest')
        
        # Determine which dataset split to use
        set_to_choose = dataset_split_handler(
            set_type=set_type, split=split, prefix=subset_name
        )
        print(f"Using dataset split: {set_to_choose}")
        self.these_idx = h5_.get(set_to_choose)[:]
        logging.info(f"Using dataset split: {set_to_choose}")
        
        # Load dictionary info for feature names and evaluation times
        dict_info_path = os.path.join(os.path.dirname(data_path), "dict_info.yaml")
        with open(dict_info_path, "r") as f:
            self.dict_info = yaml.safe_load(f)
        if use_features:
            feat_cols = self.dict_info['feat_cols']
        if use_metadata:
            metadata_cols = self.dict_info['md_cols']
        if self.return_snid:
            self.snids = h5_.get("SNID")[:]


        # Filter selected columns or use all if none specified
        if selected_features:
            self.selected_feat_index = [
                feat_cols.index(f) for f in selected_features if f in feat_cols
            ]
        else:
            if use_features:
                self.selected_feat_index = list(range(len(feat_cols)))
            
        if selected_mds:
            self.selected_md_index = [
                metadata_cols.index(m) for m in selected_mds if m in metadata_cols
            ]
        else:
            if use_metadata:
                self.selected_md_index = list(range(len(metadata_cols)))
        if use_lightcurve:
            self.mask = h5_.get("mask")
            self.data = h5_.get('brightness')
            self.time = h5_.get("time")
            self.error = h5_.get("e_brightness")

        # Initialize dataframe dictionary for tabular data
        if self.use_metadata or self.use_features:
            self.df = {}
            if self.use_metadata:
                metadata_desc = (
                    selected_mds if selected_mds is not None else 'All metadata'
                )
                logging.info(f"Using metadata columns: {metadata_desc}")
                self.df['metadata'] = h5_.get("metadata_feat")[:]
            if self.use_features:
                features_desc = (
                    selected_features if selected_features is not None 
                    else 'All features'
                )
                logging.info(f"Using feature columns: {features_desc}")
                self.df['features'] = h5_.get('extracted_feat_None')[:]
        if self.normalize_tab:
            self.load_and_process_tabular_data()
        if self.return_snid:
            if subset_name is not None:
                self.target = h5_.get(f"{subset_name}_labels")
            else:
                self.target = h5_.get("labels")
        
        # Apply filtering immediately after loading target labels
        logging.info('Loaded {} objects'.format(len(self.these_idx)))

    def load_and_process_tabular_data(self):
        if self.use_metadata:
            self.preprocess('metadata')
        if self.use_features:
            self.preprocess('features')


    def preprocess(self, data_type: str):
        """
        Preprocess the dataset features using quantile transformation.
        
        For train splits, fit a scaler and save it.
        For valid/test splits, load the scaler and transform only.
        
        Args:
            data_type (str): Type of data to preprocess ('metadata', 'features', 
                            or 'periodogram')
        """
        subset_name = self.subset_name if self.subset_name is not None else "all"
        scaler_path = (
            f"{os.path.dirname(self.data_path)}/folds/"
            f"{subset_name}_{self.sequence_type}/{data_type}/fold_{self.split}.joblib"
        )
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        

        # For training data, fit a new scaler
        if (not os.path.exists(scaler_path)) and self.set_type == "train":
            logging.info(f"Fitting scaler for {data_type} data")
            selected_data = self.df[data_type][self.these_idx]
            selected_data = np.nan_to_num(
                selected_data, nan=np.nan, posinf=np.nan, neginf=np.nan
            )
            selected_data += 1e-10
            scaler = Scaler(subsample=None,n_quantiles=1000).fit(selected_data)
            #print scaler quantiles
            dump(scaler, scaler_path)
        else:
            # For validation/test, load the existing scaler
            logging.info(f"Loading scaler for {data_type} data from {scaler_path}")
            scaler = load(scaler_path)
        # Transform the data with proper cleaning
        # Add small value to avoid log(0) issues
        data_cleaned = self.df[data_type] + 1e-10  
        data_cleaned = np.nan_to_num(
            data_cleaned, nan=np.nan, posinf=np.nan, neginf=np.nan
        )
        logging.info(f"Cleaning and transforming {data_type} data")
        data_cleaned = scaler.transform(data_cleaned) + 0.1
        # Store the processed data in the dataframe
        data_cleaned = np.nan_to_num(
            data_cleaned, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.df[data_type] = data_cleaned
        



    def order_sampling(self, idx_, at_time=None):
        """
        Extract the basic light curve data for a specific sample index.
        
        Args:
            idx_ (int): Index of the sample in the full dataset
            at_time (float, optional): Time point for evaluation
            
        Returns:
            dict: Dictionary containing the light curve data tensors
        """
        # Extract the core time series data for this index
        data = self.data[idx_,:,:]
        mask = self.mask[idx_,:,:]
        time = self.time[idx_,:,:]
        error = self.error[idx_,:,:]
        ts_data = {
            'data': data,
            'mask': mask,
            'time': time,
            'error': error
        }
        ts_data = self.extract_band_data(ts_data)
        #data_dict = self.generate_dict_(ts_data, idx_)
        data_dict = ts_data
        return data_dict
    def get_tabular_data(self,data_dict, idx_):
        tabular_mask_length = 0
        if self.use_metadata:
            metadata_data = self.df['metadata'][idx_,self.selected_md_index]
            data_dict["metadata"] = torch.from_numpy(metadata_data).float()
            data_dict['metadata_info'] = (
                np.array(self.selected_md_index).astype(np.int16) + 200 
            )
            tabular_mask_length += len(self.selected_md_index)
        if self.use_features:
            features_data = self.df['features'][idx_,self.selected_feat_index]
            # Validate features for any remaining invalid values
            data_dict["features"] = torch.from_numpy(features_data).float()
            data_dict['features_info'] = (
                np.array(self.selected_feat_index).astype(np.int16) 
            ) 
            tabular_mask_length += len(self.selected_feat_index)
        data_dict['tabular_mask'] = torch.ones(tabular_mask_length, 1)
        return data_dict
    
    def normalize_data(self,data):
        if self.norm_type is None:
            return data
        elif self.norm_type == 'arcsinh':
            data = np.arcsinh(data)
            return data
        elif self.norm_type == 'log':
            data = np.log(data + 1e-10)
        elif self.norm_type == 'zero_mean':
            data = data - np.mean(data, axis=-1, keepdims=True)
            return data

    def normalize_data(self,data):
        if self.norm_type is None:
            return data
        elif self.norm_type == 'arcsinh':
            data = np.arcsinh(data)
            return data
        elif self.norm_type == 'log':
            data = np.log(data + 1e-10)
            return data
        elif self.norm_type == 'zero_mean':
            data = data - np.mean(data, axis=-1, keepdims=True)
            return data
        else:
            return data

    def patch_sequence(self, ts_data, patch_size):
        """
        Patch the sequence data into fixed-size patches.
        
        Args:
            ts_data (dict): Dictionary containing time series data
            patch_size (int): Size of each patch
            
        Returns:
            dict: Patches of the time series data
        """
        # Calculate the number of patches needed
        num_patches = int(np.ceil(len(ts_data['data']) / patch_size))
        
        # Initialize empty arrays for patched data
        patched_data = np.zeros((num_patches, patch_size))
        patched_time = np.zeros((num_patches, patch_size))
        patched_mask = np.zeros((num_patches, patch_size), dtype=bool)
        patched_bands = np.zeros((num_patches, patch_size), dtype=int)
        patched_error = np.zeros((num_patches, patch_size))
        
        for i in range(num_patches):
            start_idx = i * patch_size
            end_idx = min(start_idx + patch_size, len(ts_data['data']))
            length = end_idx - start_idx
            
            # Fill the patches with available data
            patched_data[i, :length] = ts_data['data'][start_idx:end_idx]
            patched_time[i, :length] = ts_data['time'][start_idx:end_idx]
            patched_mask[i, :length] = ts_data['mask'][start_idx:end_idx]
            patched_bands[i, :length] = ts_data['bands'][start_idx:end_idx]
            if 'error' in ts_data:
                patched_error[i, :length] = ts_data['error'][start_idx:end_idx]
        

        ts_data['data'] = patched_data
        ts_data['time'] = patched_time
        ts_data['mask'] = patched_mask
        ts_data['bands'] = patched_bands
        #if 'error' in ts_data:
        #    ts_data['error'] = patched_error
        ###
        #remove error key if present in ts_data
        
        #print(patched_mask.shape)
        if 'error' in ts_data:
            del ts_data['error']
        return ts_data

    def generate_dict_(self,ts_data):
        data_dict = {}
        if self.patch_size > 1:
            ts_data = self.patch_sequence(ts_data, self.patch_size)
        
        for key in ts_data:
            if key == 'bands':
                # Convert band information to tensor
                data_dict[key] = torch.from_numpy(ts_data[key]).to(torch.int8)
            elif key == 'mask':
                data_dict[key] = torch.from_numpy(ts_data[key]).to(torch.bool)
            else:
                tensor_data = ts_data[key].astype(np.float32)
                data_dict[key] = torch.from_numpy(tensor_data).float()
        
        return data_dict
    def generate_dict_dino(self, ts_data):
        data_dict = {}
        views = []
        for view in ts_data['views']:
            if self.patch_size > 1:
                ts_data_view = self.patch_sequence(view, self.patch_size)
            else:
                ts_data_view = view
                
            view_dict = {}
            for key in ts_data_view:
                if key == 'bands':
                    view_dict[key] = torch.from_numpy(ts_data_view[key]).to(torch.int8)
                elif key == 'mask':
                    view_dict[key] = torch.from_numpy(ts_data_view[key]).to(torch.bool)
                else:
                    tensor_data = ts_data_view[key].astype(np.float32)
                    view_dict[key] = torch.from_numpy(tensor_data).float()
            views.append(view_dict)
        return views

    def generate_dict_tabular(self, idx_):
        data_dict = {}
        if self.use_metadata:
            metadata_data = self.df['metadata'][idx_,self.selected_md_index]
            data_dict["metadata"] = torch.from_numpy(metadata_data).float()
            data_dict['metadata_info'] = (
                np.array(self.selected_md_index).astype(np.int16) + 200 
            )
        if self.use_features:
            features_data = self.df['features'][idx_,self.selected_feat_index]
            # Validate features for any remaining invalid values
            data_dict["features"] = torch.from_numpy(features_data).float()
            data_dict['features_info'] = (
                np.array(self.selected_feat_index).astype(np.int16) 
            ) 

        data_dict["label"] = self.target[idx_]
        return data_dict
    
    def extract_band_data(self, ts_data):

        band_info = ts_data['mask'].astype(int)
        band_info[:, 0] = np.where(band_info[:, 0] == 1, 1, 0)
        band_info[:, 1] = np.where(band_info[:, 1] == 1, 2, 0)
        data_ = ts_data['data'][ts_data['mask'] == 1]
        time_ = ts_data['time'][ts_data['mask'] == 1]
        error_ = ts_data['error'][ts_data['mask'] == 1]
        mask_ = ts_data['mask'][ts_data['mask'] == 1]
        band_info_ = band_info[ts_data['mask'] == 1]

        sorting_key = time_
        sorted_indices = np.argsort(sorting_key)
        data_ = data_[sorted_indices]
        data_ = self.normalize_data(data_)
        time_ = time_[sorted_indices]
        error_ = error_[sorted_indices]
        mask_ = mask_[sorted_indices].copy()
        band_info_ = band_info_[sorted_indices]

        time_ = time_ - time_.min()
        ts_data['data'] = data_[:self.max_length]
        ts_data['time'] = time_[:self.max_length]
        ts_data['mask'] = mask_[:self.max_length]
        ts_data['bands'] = band_info_[:self.max_length]
        ts_data['error'] = error_[:self.max_length]
        return ts_data


    
    def __getitem__(self, idx):
        """
        Get a single data sample from the dataset.
        
        This method is called by PyTorch DataLoader to construct batches.
        
        Args:
            idx (int): Index in the current split (not the same as the index 
                      in the full dataset)
            
        Returns:
            dict: Dictionary containing all requested data modalities for the sample
        """
        # Map the split index to the actual index in the full dataset
        idx_ = self.these_idx[idx]
        if self.use_lightcurve:
            data_dict = self.order_sampling(idx_=idx_)
        else:
            data_dict = self.generate_dict_tabular(idx_)
    
        tsd1, tsd2 = self.transform_module(data_dict)
        if self.DINO:
            dd1 = self.generate_dict_dino(tsd1)
            return dd1
        else:
            dd1 = self.generate_dict_(tsd1)
            dd2 = self.generate_dict_(tsd2)
        if self.return_snid:
            dd1['oid'] = self.snids[idx_]
            dd2['oid'] = self.snids[idx_]
        return dd1, dd2

        
    def __len__(self):
        """
        Return the length of the dataset.
        
        Required by PyTorch Dataset interface.
        
        Returns:
            int: Number of samples in this dataset split
        """
        return len(self.these_idx)
    
    def get_labels(self):
        """
        Return all labels for the current dataset split.
        
        Returns:
            torch.Tensor: Tensor containing all class labels
        """
        return self.target[self.these_idx]
    def get_augmentations(self,idx,transform_module=None):
        """
        Return the augmentation module used for this dataset.
        
        Returns:
            object: The transformation module applied to the dataset
        """
        idx_ = self.these_idx[idx]
        if self.use_lightcurve:
            data_dict = self.order_sampling(idx_=idx_)
        else:
            data_dict = self.generate_dict_tabular(idx_)

        
        if self.DINO:
            tsd1 = transform_module(data_dict)
            dd1 = self.generate_dict_dino(tsd1)
            return dd1
        tsd1, tsd2 = transform_module(data_dict)
        dd1 =self.generate_dict_(tsd1)
        dd1['oid'] = self.snids[idx_]
        dd2 =self.generate_dict_(tsd2)
        dd2['oid'] = self.snids[idx_]
        data_dict['label'] = self.target[idx_]
        if self.use_features:
            data_dict = self.get_tabular_data(data_dict, idx_)
        return data_dict,dd1, dd2
    
