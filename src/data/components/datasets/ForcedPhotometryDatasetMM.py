"""
Forced Photometry Dataset Module for Multi-Modal astronomical time series data.

This module provides a PyTorch Dataset implementation for loading and processing 
forced photometry light curve data with multiple modalities (time series data, metadata, 
features, and periodograms).
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
from sklearn.model_selection import train_test_split



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
    else:
        if set_type == "test":
            return "test"
        elif set_type == "valid":
            return f"validation_{split}"
        elif set_type == "train":
            return f"training_{split}"
    
class ForcedPhotometryDatasetMM(Dataset):
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
        use_features=True,      # Whether to include extracted features
        eval_time=None,         # Evaluation time points for feature extraction
        normalize_tab=True,    # Whether to normalize tabular data
        mta=False,              # Whether to use multi-time augmentation
        on_evaluation=False,    # Whether the dataset is used for evaluation
        max_length=200,         # Maximum sequence length
        sampling_strategy='truncate',  # Sequence length strategy: 'truncate', 'random', 'pad_only'
        detections=False,  # Whether to use detection masks
        selected_features=None,  # Names of selected features
        selected_mds=None,  # Names of selected metadata columns
        min_detections=0,
        sequence_type='windows',  # Minimum number of detections for filtering
        max_time_to_eval=None,  # Maximum time to evaluate
        norm_type=None,  # Normalization type for data
        percentage=1.0,  # Percentage of data to use
        return_snids=False,  # Whether to return SNIDs
        patch_size=0,  # Patch size for band patching
        regression=False,
        corrected_periods=False,
        double_augmentation=None,
        **kwargs,
    ):
        """
        Initialize the Forced Photometry Dataset.
        """
        # Store configuration parameters
        self.patch_size = patch_size
        self.max_length = max_length
        self.sampling_strategy = sampling_strategy
        self.use_lightcurve = use_lightcurve
        self.dataset_dir = data_dir
        self.eval_time = eval_time
        self.normalize_tab = normalize_tab
        self.mta = mta
        self.on_evaluation = on_evaluation
        self.split = split
        self.set_type = set_type
        self.sequence_type = sequence_type
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.subset_name = subset_name
        self.max_time_to_eval = max_time_to_eval
        self.norm_type = norm_type
        self.return_snids = return_snids
        self.regression = regression
        if self.regression and selected_features is None:
            self.use_features = False
        # Set up data paths and load file
        data_path = os.path.join(data_dir, data_name)+'_'+sequence_type+".h5"
        self.data_path = data_path
        h5_ = h5py.File(data_path, "r")
        
        # Determine which dataset split to use
        set_to_choose = dataset_split_handler(set_type=set_type, split=split, prefix=subset_name)
        self.these_idx = h5_.get(set_to_choose)[:]
        logging.info(f"Using dataset split: {set_to_choose}")
        
        # Load dictionary info for feature names and evaluation times
        dict_info_path = os.path.join(os.path.dirname(data_path), "dict_info.yaml")
        with open(dict_info_path, "r") as f:
            self.dict_info = yaml.safe_load(f)
            
        feat_cols = self.dict_info['feat_cols']
        metadata_cols = self.dict_info['md_cols']
        if return_snids:
            self.snids = h5_.get("SNID")


        # Filter selected columns or use all if none specified
        self.selected_feat_index = [feat_cols.index(f) for f in selected_features if f in feat_cols] if selected_features else list(range(len(feat_cols)))
        self.selected_md_index = [metadata_cols.index(m) for m in selected_mds if m in metadata_cols] if selected_mds else list(range(len(metadata_cols)))
        if use_lightcurve:
            
            logging.info(f"Number of samples = {len(self.these_idx)}")
            if min_detections > 0:
                self.mask_detection = h5_.get("mask_detection")
                self.these_idx, self.banned_idx = self.apply_detection_filter(
                    dataset_indices=self.these_idx,
                    mask=self.mask_detection,
                    min_detections=min_detections
                )
                logging.info(f'Removed {len(self.banned_idx)} samples due to insufficient detections')
                logging.info(f"After filtering by detections: {len(self.these_idx)} samples remain")
            self.mask = h5_.get("mask") if not detections else self.mask_detection
            if detections:
                logging.info(f"Using detection mask")
            self.data = h5_.get('brightness')
            self.time = h5_.get("time")
        #self.apply_max_time_to_eval()

        if self.use_metadata or self.use_features:
            self.df = {}
            if self.use_metadata:
                logging.info(f"Using metadata columns: {selected_mds if selected_mds is not None else 'All metadata'}")
                self.df['metadata'] = h5_.get("metadata_feat")[:]
            if self.use_features:
                logging.info(f"Using feature columns: {selected_features if selected_features is not None else 'All features'}")
                self.df['features'] = h5_.get('extracted_feat_None')[:] if (self.max_time_to_eval is None or self.max_time_to_eval==2048) else h5_.get(f'extracted_feat_{self.max_time_to_eval}')[:]
                for t in self.eval_time:
                     self.df[f"features_{t}"] = h5_.get(f"extracted_feat_{t}")[:]

        if self.normalize_tab:
            self.load_and_process_tabular_data()
        
        if self.regression:
            if corrected_periods:
                self.target = h5_.get("corrected_period")[:,0]
                self.target = self.fix_periods_values(self.target)
        else:
            self.target = h5_.get(f"{subset_name}_labels") if subset_name is not None else h5_.get("labels")
        
        if percentage < 1.0 and not self.on_evaluation:
            y_labels = self.target[self.these_idx]
            
            # Apply percentage-based filtering if specified
            these_idx, _ = train_test_split(
                self.these_idx,
                train_size=percentage,
                stratify=y_labels,
                test_size=1 - percentage,
                random_state=40,
                shuffle=True
            )
            self.these_idx = np.sort(these_idx)
    def fix_periods_values(self,period_values):
        # Replace negative or zero period values with 1
        period_values = np.where(period_values <= 0, 1.0, period_values)
        return period_values
        
        # Close the HDF5 file
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
        scaler_path = f"{os.path.dirname(self.data_path)}/folds/{subset_name}_{self.sequence_type}/{data_type}/fold_{self.split}.joblib"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        

        # For training data, fit a new scaler
        if (not os.path.exists(scaler_path)) and self.set_type == "train":
            logging.info(f"Fitting scaler for {data_type} data")
            selected_data = self.df[data_type][self.these_idx]
            selected_data = np.nan_to_num(selected_data, nan=np.nan, posinf=np.nan, neginf=np.nan)
            selected_data += 1e-10
            scaler = Scaler(subsample=None,n_quantiles=1000).fit(selected_data)
            #print scaler quantiles
            dump(scaler, scaler_path)
        else:
            # For validation/test, load the existing scaler
            logging.info(f"Loading scaler for {data_type} data from {scaler_path}")
            scaler = load(scaler_path)
        # Transform the data with proper cleaning
        data_cleaned = self.df[data_type] + 1e-10  # Add small value to avoid log(0) issues
        data_cleaned = np.nan_to_num(data_cleaned, nan=np.nan, posinf=np.nan, neginf=np.nan)
        logging.info(f"Cleaning and transforming {data_type} data")
        data_cleaned = scaler.transform(data_cleaned) + 0.1
        # Store the processed data in the dataframe
        data_cleaned = np.nan_to_num(data_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        self.df[data_type] = data_cleaned
        
        # For features, also transform the time-specific feature lists if they exist
        if data_type == 'features' and self.eval_time is not None and self.set_type == "train" and self.mta:
            for t in self.eval_time:
                    print(f"Processing features for time {t}")
                    data_cleaned = self.df[f"features_{t}"]
                    data_cleaned = np.nan_to_num(data_cleaned, nan=np.nan, posinf=np.nan, neginf=np.nan)
                    # Clean the data before transforming
                    data_cleaned = scaler.transform(data_cleaned) + 0.1
                    data_cleaned = np.nan_to_num(data_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
                    self.df[f"features_{t}"] = data_cleaned


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

        ts_data = {
            'data': data,
            'mask': mask,
            'time': time
        }
        ts_data = self.extract_band_data(ts_data)
        # Set the sequence length for the data
        ts_data = self.set_sequence_length(ts_data, self.sampling_strategy)
        if self.patch_size > 1:
            # Apply band patching if patch size is greater than 1
            ts_data = self.patch_sequence(ts_data, self.patch_size)
        data_dict = self.generate_dict_(ts_data, idx_)
        
        return data_dict
    def patch_sequence(self, ts_data, patch_size):
        """
        Patch the sequence data into fixed-size patches in temporal order.
        
        Args:
            ts_data (dict): Dictionary containing time series data
            patch_size (int): Size of each patch
            
        Returns:
            dict: Patches of the time series data in temporal order
        """
        data = ts_data['data']
        time = ts_data['time']
        mask = ts_data['mask']
        bands = ts_data['bands']
    
        # Calculate number of patches needed
        num_patches = int(np.ceil(len(data) / patch_size))
        
        # Initialize patches
        patched_data = np.zeros((num_patches, patch_size))
        patched_time = np.zeros((num_patches, patch_size))
        patched_mask = np.zeros((num_patches, patch_size), dtype=bool)
        patched_bands = np.zeros((num_patches, patch_size), dtype=int)
        
        for i in range(num_patches):
            start_idx = i * patch_size
            end_idx = min(start_idx + patch_size, len(data))
            length = end_idx - start_idx
        
            # Fill the patches with available data in temporal order
            patched_data[i, :length] = data[start_idx:end_idx]
            patched_time[i, :length] = time[start_idx:end_idx]
            patched_mask[i, :length] = mask[start_idx:end_idx]
            patched_bands[i, :length] = bands[start_idx:end_idx]
    
        return {
            'data': patched_data,
            'time': patched_time,
            'mask': patched_mask,
            'bands': patched_bands
        }
    def get_tabular_data(self,data_dict, idx_):
        tabular_mask_length = 0
        if self.use_metadata:
            metadata_data = self.df['metadata'][idx_,self.selected_md_index]
            data_dict["metadata"] = torch.from_numpy(metadata_data).float()
            data_dict['metadata_info'] = np.array(self.selected_md_index).astype(np.int16) + 200 
            tabular_mask_length += len(self.selected_md_index)
        if self.use_features:
            features_data = self.df['features'][idx_,self.selected_feat_index]
            # Validate features for any remaining invalid values
            data_dict["features"] = torch.from_numpy(features_data).float()
            data_dict['features_info'] = np.array(self.selected_feat_index).astype(np.int16) 
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

    def generate_dict_(self,ts_data, idx_):
        data_dict = {}

        for key in ts_data:
            if key == 'bands':
                # Convert band information to tensor
                data_dict[key] = torch.from_numpy(ts_data[key]).to(torch.int8)
            elif key == 'mask':
                # Convert mask to float tensor
                data_dict[key] = torch.from_numpy(ts_data[key]).to(torch.bool)
            else:
                data_dict[key] = torch.from_numpy(ts_data[key]).float()
        
        data_dict = self.get_tabular_data(data_dict, idx_)
        data_dict["label"] = self.target[idx_]
        if self.regression:
            data_dict["label"] = np.log10(self.target[idx_])
        return data_dict

    def generate_dict_tabular(self, idx_):
        data_dict = {}
        if self.use_metadata:
            metadata_data = self.df['metadata'][idx_,self.selected_md_index]
            data_dict["metadata"] = torch.from_numpy(metadata_data).float()
            data_dict['metadata_info'] = np.array(self.selected_md_index).astype(np.int16) + 200 
        if self.use_features:
            features_data = self.df['features'][idx_,self.selected_feat_index]
            # Validate features for any remaining invalid values
            data_dict["features"] = torch.from_numpy(features_data).float()
            data_dict['features_info'] = np.array(self.selected_feat_index).astype(np.int16) 

        data_dict["label"] = self.target[idx_]
        return data_dict
    
    def extract_band_data(self, ts_data):

        band_info = ts_data['mask'].astype(int)
        band_info[:, 0] = np.where(band_info[:, 0] == 1, 1, 0)
        band_info[:, 1] = np.where(band_info[:, 1] == 1, 2, 0)
        data_ = ts_data['data'][ts_data['mask'] == 1]
        time_ = ts_data['time'][ts_data['mask'] == 1]
        mask_ = ts_data['mask'][ts_data['mask'] == 1]
        band_info_ = band_info[ts_data['mask'] == 1]

        sorting_key = time_
        sorted_indices = np.argsort(sorting_key)
        data_ = data_[sorted_indices]
        data_ = self.normalize_data(data_)
        time_ = time_[sorted_indices]
        mask_ = mask_[sorted_indices]
        band_info_ = band_info_[sorted_indices]

        time_ = time_ - time_.min()
        ts_data['data'] = data_
        ts_data['time'] = time_
        ts_data['mask'] = mask_
        ts_data['bands'] = band_info_
        return ts_data


    def set_sequence_length(self, ts_dict, sampling_strategy='truncate'):
        """
        Set sequence length to max_length using specified strategy.
        
        Args:
            ts_dict (dict): Time series data dictionary
            sampling_strategy (str): 'truncate', 'random', or 'pad_only'
                - 'truncate': Take first max_length points (original behavior)
                - 'random': Randomly sample max_length points
                - 'pad_only': Only pad if sequence is shorter, don't truncate
        """
        if len(ts_dict['data']) > self.max_length:

            if sampling_strategy == 'random' and not self.on_evaluation:
                ts_data = self._random_sample_sequence(ts_dict, self.max_length)
            else:
                ts_data = self._truncate_sequence(ts_dict, self.max_length)
        elif len(ts_dict['data']) < self.max_length:
            ts_data = self._pad_sequence(ts_dict, self.max_length)
        else:  # len(ts_dict['data']) == self.max_length
            ts_data = ts_dict
        return ts_data
    def _truncate_sequence(self, ts_data, max_length):
            """Truncate sequence to max_length."""
            result = {}
            for key in ts_data:
                result[key] = ts_data[key][:max_length]
            return result
    
    def _pad_sequence(self, ts_data, max_length):
        """Pad sequence to max_length."""
        result = {}
        actual_len = len(ts_data['data'])
        padding_len = max_length - actual_len
        
        # Define padding for each key
        padding_values = {
            'data': 0,
            'time': 0,
            'mask': 0,
            'bands': 0,
        }
        
        for key in ts_data:
            pad_value = padding_values.get(key, 0)
            result[key] = np.pad(ts_data[key], ((0, padding_len),), 'constant', constant_values=pad_value)
        
        return result

    def three_time_mask(self, data_dict, idx_):
        """
        Apply a time-based mask to simulate early light curve observations.
        
        This is used for Multi-Time Augmentation (MTA) during training to simulate
        the model seeing the light curve at different stages of observation.
        
        Args:
            data_dict (dict): Dictionary containing the light curve data
            idx_ (int): Index of the sample in the full dataset
            
        Returns:
            dict: Updated data dictionary with time-masked observations
        """
        # Get the mask and time arrays from the data dict
        mask, time = data_dict['mask'], data_dict['time']
        
        # Randomly select an evaluation time poin
        eval_time = np.random.choice(self.eval_time)
        eval_time_dict = {
            'None': None,
            '8': 8.0,
            '16': 16.0,
            '32': 32.0,
            '64': 64.0,
            '128': 128.0,
            '256': 256.0,
            '512': 512.0,
            '1024': 1024.0,
        }
        eval_time_num = eval_time_dict.get(eval_time, None)
        if self.use_lightcurve and eval_time_num is not None:
            # Create a mask to only keep observations before eval_time
                #transfor eval_time to tensor with dtype float
                mask_time = (time <= eval_time_num)
                # Apply the time-based mask to data, mask and time
                data_dict['mask'] = torch.where(mask_time, data_dict['mask'], torch.zeros_like(data_dict['mask']))
                data_dict['time'] = torch.where(mask_time, data_dict['time'], torch.zeros_like(data_dict['time']))
                data_dict['data'] = torch.where(mask_time, data_dict['data'], torch.zeros_like(data_dict['data']))
                data_dict['bands'] = torch.where(mask_time, data_dict['bands'], torch.zeros_like(data_dict['bands']))

            # Add the time-specific feature vector if using features
        if self.use_features:
            data_dict['features'] = torch.from_numpy(self.df[f"features_{eval_time}"][idx_]).float()

        return data_dict
    def cut_to_max_time(self, data_dict):
        """
        Cut the light curve data to the maximum time specified for evaluation.
        
        This method is used to ensure that the light curve data does not exceed
        a certain time limit, which is useful for evaluation purposes.
        
        Args:
            data_dict (dict): Dictionary containing the light curve data
            
        Returns:
            dict: Updated data dictionary with cut light curve data
        """
        if self.max_time_to_eval is None or self.sequence_type == 'windows' or self.max_time_to_eval >= 2047:
            return data_dict
        
        # Get the time data and apply the cut
        time_data = data_dict['time']
        min_time = time_data.min()
        cut = min_time + self.max_time_to_eval
        
        # Apply the cut to all relevant fields in the data_dict
        for key in ['data', 'mask', 'time', 'bands']:
            data_dict[key] = torch.where(time_data <= cut, data_dict[key], torch.zeros_like(data_dict[key]))
        
        return data_dict
    
    def __getitem__(self, idx):
        """
        Get a single data sample from the dataset.
        
        This method is called by PyTorch DataLoader to construct batches.
        
        Args:
            idx (int): Index in the current split (not the same as the index in the full dataset)
            
        Returns:
            dict: Dictionary containing all requested data modalities for the sample
        """
        # Map the split index to the actual index in the full dataset
        idx_ = self.these_idx[idx]
        if self.use_lightcurve:
        # Get the base light curve data
            data_dict = self.order_sampling(idx_=idx_)
        else:
            # If not using light curves, just get the tabular data
            data_dict = self.generate_dict_tabular(idx_)
        if self.mta and not self.on_evaluation:
            # Apply time-based augmentation if specified
            data_dict = self.three_time_mask(data_dict, idx_)
        if self.on_evaluation:
            if self.return_snids:
                data_dict['oid'] = self.snids[idx_]
            data_dict = self.cut_to_max_time(data_dict)
        return data_dict
    
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
    
    def _random_sample_sequence(self, ts_data, max_length):
        """
        Randomly sample max_length points from the sequence.
        
        This function randomly selects a subset of data points from the time series
        while maintaining temporal order. This can be useful for data augmentation
        or when you want to work with a fixed-size subset of observations.
        
        Args:
            ts_data (dict): Time series data dictionary containing 'data', 'time', 
                           'mask', and 'bands' arrays
            max_length (int): Maximum number of points to sample
            
        Returns:
            dict: Dictionary with the same keys but with randomly sampled data points
        """
        actual_len = len(ts_data['data'])
        
        if actual_len <= max_length:
            # If sequence is already shorter or equal, return as is
            return ts_data
        # Randomly select indices without replacement
        random_indices = np.random.choice(actual_len, size=max_length, replace=False)
        # Sort indices to maintain temporal order
        random_indices = np.sort(random_indices)
        
        result = {}
        for key in ts_data:
            result[key] = ts_data[key][random_indices]
        
        return result

    def filter_indices_by_detections(mask, min_detections):
        """
        Filter dataset indices based on minimum number of detections.
        
        This function identifies which samples (indices) have at least the specified
        minimum number of detections across all bands, useful for quality control
        in astronomical time series datasets.
        
        Args:
            mask (np.ndarray): Detection mask array of shape (n_samples, n_timesteps, n_bands)
                              where 1 indicates a detection and 0 indicates no detection
            min_detections (int): Minimum number of detections required per sample
            
        Returns:
            np.ndarray: Array of valid indices that have at least min_detections
            np.ndarray: Array of banned indices that have fewer than min_detections
            
        Example:
            >>> mask = np.array([[[1, 0], [1, 1], [0, 1]],  # Sample 0: 4 detections
            ...                  [[1, 1], [0, 0], [1, 0]]])  # Sample 1: 3 detections  
            >>> valid_idx, banned_idx = filter_indices_by_detections(mask, min_detections=4)
            >>> print(f"Valid indices: {valid_idx}")    # [0]
            >>> print(f"Banned indices: {banned_idx}")  # [1]
        """
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Count total detections per sample (sum across timesteps and bands)
        detections_per_sample = np.sum(mask, axis=(1, 2))
        
        # Find indices that meet the minimum detection requirement
        valid_indices = np.where(detections_per_sample >= min_detections)[0]
        banned_indices = np.where(detections_per_sample < min_detections)[0]
        
        logging.info(f"Filtering by minimum detections: {min_detections}")
        logging.info(f"Valid samples: {len(valid_indices)} out of {len(mask)}")
        logging.info(f"Banned samples: {len(banned_indices)} out of {len(mask)}")
        
        return valid_indices, banned_indices


    def apply_detection_filter(self,dataset_indices, mask, min_detections):
        """
        Apply detection filtering to an existing set of dataset indices.
        
        This function takes a subset of indices (e.g., from a train/valid/test split)
        and further filters them based on detection count requirements.
        
        Args:
            dataset_indices (np.ndarray): Original indices to filter (e.g., from train/valid/test split)
            mask (np.ndarray): Detection mask array of shape (n_samples, n_timesteps, n_bands)
            min_detections (int): Minimum number of detections required per sample
            
        Returns:
            np.ndarray: Filtered indices that both exist in dataset_indices AND meet detection requirements
            np.ndarray: Indices from dataset_indices that were removed due to insufficient detections
            
        Example:
            >>> # Original train split indices
            >>> train_indices = np.array([0, 1, 2, 3, 4])
            >>> mask = np.random.randint(0, 2, (10, 100, 2))  # 10 samples, 100 timesteps, 2 bands
            >>> filtered_train, removed_train = apply_detection_filter(train_indices, mask, min_detections=20)
        """
        if not isinstance(dataset_indices, np.ndarray):
            dataset_indices = np.array(dataset_indices)
        
        # Get the mask subset for only the given indices
        mask_subset = mask[dataset_indices]
        
        # Count detections for each sample in the subset
        detections_per_sample = np.sum(mask_subset, axis=(1, 2))
        
        # Find which samples in the subset meet the requirement
        valid_mask = detections_per_sample >= min_detections
        
        # Map back to original indices
        filtered_indices = dataset_indices[valid_mask]
        removed_indices = dataset_indices[~valid_mask]
        
        logging.info(f"Applied detection filter to {len(dataset_indices)} indices")
        logging.info(f"Kept: {len(filtered_indices)}, Removed: {len(removed_indices)}")
        
        return filtered_indices, removed_indices

    def apply_max_time_to_eval(self):
        if self.max_time_to_eval is None or self.sequence_type != 'windows' or self.max_time_to_eval == 2048:
            logging.info("No max_time_to_eval specified, skipping time-based filtering.")
            return
        print(f"Applying max_time_to_eval: {self.max_time_to_eval} seconds")
        print('Actual number of samples before filtering:', len(self.these_idx))
        selected_snids = self.snids[self.these_idx]
        #group snid by base snid
        snid_groups = {}
        for snid,idx in zip(selected_snids,self.these_idx):
            base_snid = snid.decode('utf-8').split('_')[0]  # Assuming snid is formatted like 'snid_123'
            if base_snid not in snid_groups:
                snid_groups[base_snid] = []
            snid_groups[base_snid].append(idx)
        indices_to_keep = []
        data_original = self.data[:]
        mask_original = self.mask[:]
        time_original = self.time[:]
        for base_snid, indices in snid_groups.items():
            # Get the time data for this group
            time_data = self.time[indices]
            time_data = time_data[time_data > 0]  # Filter out any zero or negative times
            min_time = np.min(time_data)
            cut = min_time + self.max_time_to_eval
            for idx in indices:
                data_original[idx] = np.where(self.time[idx] <= cut, self.data[idx], 0)
                mask_original[idx] = np.where(self.time[idx] <= cut, self.mask[idx], 0)
                time_original[idx] = np.where(self.time[idx] <= cut, self.time[idx], 0)
                # Check if there are valid non-zero time values after filtering
                if np.any(time_original[idx] > 0):
                    indices_to_keep.append(idx)
        self.data = data_original
        self.mask = mask_original
        self.time = time_original
        # Update the indices to keep only those that have valid time data

        self.these_idx = np.array(indices_to_keep).astype(int)
        print(f"Actual number of samples after filtering: {len(self.these_idx)}")
        
    def get_data_dict(self, idx):
        """
        Get a single data sample from the dataset.
        
        This method is called by PyTorch DataLoader to construct batches.
        
        Args:
            idx (int): Index in the current split (not the same as the index in the full dataset)
            
        Returns:
            dict: Dictionary containing all requested data modalities for the sample
        """
        # Map the split index to the actual index in the full dataset

        data_dict = self.order_sampling_without_pad(idx_=idx)
        return data_dict

    def order_sampling_without_pad(self, idx_, at_time=None):
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
        ts_data = {
            'data': data,
            'mask': mask,
            'time': time
        }
        ts_data = self.extract_band_data_no_norm(ts_data)

        
        return ts_data
    
    def extract_band_data_no_norm(self, ts_data):

        band_info = ts_data['mask'].astype(int)
        band_info[:, 0] = np.where(band_info[:, 0] == 1, 1, 0)
        band_info[:, 1] = np.where(band_info[:, 1] == 1, 2, 0)
        data_ = ts_data['data'][ts_data['mask'] == 1]
        time_ = ts_data['time'][ts_data['mask'] == 1]
        mask_ = ts_data['mask'][ts_data['mask'] == 1]
        band_info_ = band_info[ts_data['mask'] == 1]

        sorting_key = time_
        sorted_indices = np.argsort(sorting_key)
        data_ = data_[sorted_indices]
        time_ = time_[sorted_indices]
        mask_ = mask_[sorted_indices]
        band_info_ = band_info_[sorted_indices]
        ts_data['data'] = data_
        ts_data['time'] = time_
        ts_data['mask'] = mask_
        ts_data['bands'] = band_info_
        return ts_data

