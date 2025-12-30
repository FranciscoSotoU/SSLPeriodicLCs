# Importar datos de ztf
import os
import h5py
import numpy as np
from pathlib import Path

# Create 5-fold cross-validation partitions from train+val in a stratified way
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Optional, Tuple, Union
from joblib import dump, load
from sklearn.preprocessing import QuantileTransformer as Scaler
from sklearn.utils.class_weight import compute_class_weight
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FullTimeSeriesDataset(Dataset):
    """Dataset for time series data grouped by object ID with full light curves."""
    
    # Class label definitions for plotting and evaluation
    class_names = [
        'CV/Nova', 'LPV', 'YSO', 'QSO', 'RSCVn', 'CEP', 'EA', 'RRLab', 'RRLc', 
        'SNIa', 'SNII', 'AGN', 'EB/EW', 'DSCT', 'Blazar', 'Microlensing', 
        'SNIIn', 'Periodic-Other', 'SNIbc', 'SLSN', 'TDE', 'SNIIb'
    ]

    class_ordered = [
        "SNIa", "SNIbc", "SNIIb", "SNII", "SNIIn", "SLSN", "TDE", "Microlensing",
        "QSO", "AGN", "Blazar", "YSO", "CV/Nova", "LPV", "EA", "EB/EW",
        "Periodic-Other", "RSCVn", "CEP", "RRLab", "RRLc", "DSCT"
    ]
    
    # Class hierarchy for visualization grouping
    class_hierarchy = {
        'Transient': ['SLSN', 'SNII', 'SNIa', 'SNIbc', 'TDE', 'SNIIb', 'SNIIn', 'Microlensing'],
        'Stochastic': ['QSO', 'YSO', 'AGN', 'Blazar', 'CV/Nova'],
        'Periodic': ['CEP', 'DSCT', 'EA', 'RRLc', 'LPV', 'RSCVn', 'EB/EW', 'RRLab', 'Periodic-Other']
    }
    
    indices: list = None  # List of indices to use in the partition
    dataset_path: str = './data/ztf/'  # Directory where the dataset is stored
    set_type: str = "train"  # Set type
    force_online_opt: bool = True  # Force online optimization for random mask
    online_opt_tt: bool = True  # Online optimization for three time points
    normalize_ts: bool = True  # Normalize time series data
    normalize_tab: bool = True  # Normalize tabular data
    use_metadata: bool = True  # Use metadata 
    use_features: bool = False  # Use features
    per_init_time: float = 0.2  # Initial time % for random mask
    max_time: int = 1500  # Maximum time for random mask
    force_qt: bool = False  # Force the Scalers to be fitted to the training data
    fold: int = 0  # Fold used for this experiment
    verbose: bool = True  # Print verbose output
    dataset: str = "ztf"  # Dataset name
    collate: bool = True  # Collate the data for the dataloader
    return_hierarchy: bool = False  # Return the class hierarchy
    max_samples: int = 2048  # Maximum number of time points to sample (when not using full LC)
    max_len: int = 2048  # Maximum sequence length for padding/truncating
    use_full_lc: bool = True  # Use full light curve instead of sampling
    use_random_sampling: bool = False  # Use random sampling instead of sorting by time
    kwargs: dict = field(default_factory=dict)  # Additional arguments

    def __post_init__(self, **kwargs):
        """Initialize the dataset."""
        self.kwargs.update(kwargs)

        # Load dataset
        raw = h5py.File(f"{self.dataset_path}/dataset.h5", 'r')
        self.df = {key: raw[key] for key in raw.keys()}  # Load the entire dataset pointers
        self.keys = {'data', 'mask', 'time', 'mask_photometry'}
        self.eval_time = np.array([f"{2**i}" for i in range(4 if self.dataset == "ztf" else 3, 12)])  # 16-2048 ZTF, 8-2048 ELAsTiCC
        
        # Set indices to use
        self.iloc = self.indices if self.indices is not None else np.arange(len(self.df['SNID']))
        
        # Group indices by OID
        self.oid_groups = self._group_by_oid()
        self.oid_list = list(self.oid_groups.keys())
        
        # Get labels for OIDs
        self.oid_labels = []
        self.oid_class_names = []  # Store class names
        self.oid_hierarchy_types = []  # Store hierarchy types
        
        for oid in self.oid_list:
            idx = self.oid_groups[oid][0]  # First index for this OID
            label = np.asarray(self.df['labels'][idx]).reshape(-1)[0]
            self.oid_labels.append(label)
            
            # Get class name and hierarchy type
            if 0 <= label < len(self.class_names):
                class_name = self.class_names[label.astype(int)]
                self.oid_class_names.append(class_name)

        self.oid_labels = np.array(self.oid_labels)
        
        # Log dataset info
        if self.verbose:
            logger.info(f"Initialized {self.set_type} dataset with {len(self.iloc)} samples from {len(self.oid_list)} unique objects")
            class_counts = np.bincount(self.oid_labels.flatten().astype(int))
            logger.info(f"Class distribution (by OID): {class_counts}")
  
        # Load and preprocess tabular data
        self.load_and_preprocess_tabular_data()
        
        # Calculate class weights for balanced sampling
        self._calculate_class_weights()
        
    def _group_by_oid(self) -> Dict[str, List[int]]:
        """Group indices by object ID."""
        oid_groups = defaultdict(list)
        snids = self.df['SNID'][:][self.iloc]
        
        for idx, snid in zip(self.iloc, snids):
            oid = snid.decode('utf-8').split('_')[0]
            oid_groups[oid].append(idx)
            
        return dict(oid_groups)
    
    def load_and_preprocess_tabular_data(self):
        """Load and preprocess tabular data."""
        if self.use_metadata:
            self.df['metadata'] = self.df["metadata_feat"][:]
            self.preprocess('metadata')
            
        if self.use_features:
            self.df['features'] = self.df["features_2048"][:]
            self.features_list = {f"time_{t}": self.df[f"features_{t}"][:] for t in self.eval_time}
            self.preprocess('features')
    
    def preprocess(self, data_type: str):
        """Preprocess the tabular data."""
        self.keys.add(data_type)
        if self.normalize_tab:
            scaler_path = f"{self.dataset_path}/folds/{data_type}/fold_{self.fold}.joblib"
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            
            if (not os.path.exists(scaler_path) or self.force_qt) and self.set_type == "train":
                # For training, use full dataset indices to fit the scaler
                if data_type == 'metadata':
                    scaler = Scaler().fit(self.df[data_type][self.iloc])
                elif data_type == 'features':
                    scaler = Scaler().fit(self.df[data_type][self.iloc])
                dump(scaler, scaler_path)
            else:
                scaler = load(scaler_path)
            
            self.df[data_type] = scaler.transform(self.df[data_type])
            
            # Also transform the features at different times if using features
            if data_type == 'features':
                self.features_list = {f"time_{t}": scaler.transform(self.features_list[f"time_{t}"]) for t in self.eval_time}
    
    def _calculate_class_weights(self):
        """Calculate weights for each OID based on class distribution for balanced sampling."""
        # Count occurrences of each class
        unique_labels, counts = np.unique(self.oid_labels, return_counts=True)
        
        # Calculate inverse frequency weights
        class_weights = 1.0 / counts
        
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights) * len(unique_labels)
        
        # Create a mapping from label values to indices
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        
        # Create weight for each OID based on its class
        self.weights = np.zeros(len(self.oid_list))
        for i, label in enumerate(self.oid_labels):
            # Use the mapping to find the correct index in class_weights
            self.weights[i] = class_weights[label_to_index[label]]
        
        # Convert to tensor
        self.weights = torch.FloatTensor(self.weights)
    
    def __len__(self) -> int:
        """Return the number of OIDs."""
        return len(self.oid_list)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get a sample for the given OID index."""
        # Store current sample index for augmentation
        self.current_sample_idx = idx
        
        # Get the OID and its indices
        oid = self.oid_list[idx]
        indices = self.oid_groups[oid]
        
        # Step 1: Collect time series data from all chunks
        ts_data = self._collect_chunks_for_oid(oid, indices)
        
        # Step 2: Sort and process the time series data
        ts_data = self._process_time_series(ts_data)
        
        # Step 3: Adjust sequence length (truncate/pad)
        ts_data = self._adjust_sequence_length(ts_data)
        
        # Step 4: Create the sample dictionary with all fields
        sample = self._create_sample_dict(ts_data, indices[0])
        
        # Step 5: Apply masking for training if needed
        if self.set_type == "train":
            sample = self._apply_training_masks(sample, indices[0])
        
        # Step 6: Convert all arrays to tensors
        sample = self._convert_to_tensors(sample)
        
        # Get label and convert to tensor
        label = torch.from_numpy(np.asarray(self.df['labels'][indices[0]]).reshape(-1)).long()
        
        # Apply collate function if needed
        if self.collate:
            sample, label = self.collate_fn(sample, label)
            
        return sample, label

    def _collect_chunks_for_oid(self, oid: str, indices: list) -> dict:
        """Collect all time series data from chunks for a given OID."""
        all_data = []
        all_time = []
        all_mask = []
        all_var = []
        all_bands = []
        all_chunk = []
        all_photometry_mask = []
        
        # Track maximum time seen for offsets
        max_time_so_far = 0
        time_offset = 0
        
        for i, chunk_idx in enumerate(indices):
            # Extract data for this chunk
            data = self.df['data'][chunk_idx]
            time = self.df['time'][chunk_idx]
            mask = self.df['mask'][chunk_idx]
            data_var = self.df['data-var'][chunk_idx]
            chunk_id = np.ones_like(time) * i
            photometry_mask = self.df['mask_photometry'][chunk_idx]
            
            # Apply normalization if needed
            if self.normalize_ts:
                data = np.arcsinh(data)
            
            # Process each band in this chunk
            max_time_so_far = self._extract_band_data(
                data, time, mask, data_var, chunk_id, photometry_mask,
                all_data, all_time, all_mask, all_var, all_bands, all_chunk, all_photometry_mask,
                i, time_offset, max_time_so_far
            )
            
            # Update time offset for next chunk
            if i < len(indices) - 1:
                time_offset = max_time_so_far
        
        return {
            'data': all_data,
            'time': all_time,
            'mask': all_mask,
            'var': all_var,
            'bands': all_bands,
            'chunk': all_chunk,
            'photometry_mask': all_photometry_mask
        }

    def _extract_band_data(self, data, time, mask, data_var, chunk_id, photometry_mask,
                           all_data, all_time, all_mask, all_var, all_bands, all_chunk, all_photometry_mask,
                           chunk_index, time_offset, max_time_so_far):
        """Extract data for each band and apply time offsets."""
        n_times, n_bands = data.shape
        
        for band_idx in range(n_bands):
            # Extract data for this band
            band_data = data[:, band_idx]
            band_time = time[:, band_idx]
            band_mask = mask[:, band_idx]
            band_var = data_var[:, band_idx]
            band_chunk = chunk_id[:, band_idx]
            band_photometry_mask = photometry_mask[:, band_idx]
            
            # Only keep valid points (mask > 0)
            valid_mask = band_mask > 0
            
            if np.any(valid_mask):  # Only add if there are valid points
                # Extract the valid data points
                valid_data = band_data[valid_mask]
                valid_time = band_time[valid_mask]
                valid_mask_data = band_mask[valid_mask]
                valid_var = band_var[valid_mask]
                valid_chunk = band_chunk[valid_mask]
                valid_photometry_mask = band_photometry_mask[valid_mask]
                
                # Apply time offset to chunks after the first
                if chunk_index > 0 and len(valid_time) > 0:
                    valid_time = valid_time + time_offset
                    
                # Update max time for next chunk
                if len(valid_time) > 0:
                    current_max = np.max(valid_time)
                    max_time_so_far = max(max_time_so_far, current_max)
                
                # Add to collections
                all_data.append(valid_data)
                all_time.append(valid_time)
                all_mask.append(valid_mask_data)
                all_var.append(valid_var)
                all_chunk.append(valid_chunk)
                all_photometry_mask.append(valid_photometry_mask)
                
                # Create band indicators
                band_indicators = np.full(np.sum(valid_mask), band_idx)
                all_bands.append(band_indicators)
        
        # Return the updated max_time
        return max_time_so_far

    def _process_time_series(self, ts_data):
        """Process collected time series data - concatenate and sort."""
        # Handle empty case
        if not ts_data['data']:
            # Create minimal dummy data
            return self._create_dummy_data()
        
        # Concatenate all collected arrays
        for key in ts_data:
            ts_data[key] = np.concatenate(ts_data[key])
        sorting_key = ts_data['time'] + (1 - ts_data['mask']) * 1e9
        time_order = np.argsort(sorting_key)
        
        # Apply sorting
        for key in ts_data:
            ts_data[key] = ts_data[key][time_order]
        
        return ts_data

    def mask_data(self, sample: dict, mask_type: str, oid_idx: int) -> dict:
        """Mask the data based on the mask type."""
        mask, time_alert = sample["mask"], sample["time"]
        eval_time = self._get_eval_time(mask_type, time_alert, mask)
        mask_time = (time_alert <= eval_time).astype(float)
        sample["mask"] = mask * mask_time
        
        if self.use_features:
            sample["features"] = self.features_list[
                f"time_{self.eval_time[(eval_time <= self.eval_time.astype(int)).argmax()]}"
            ][oid_idx, :]
        return sample
    
    def _get_eval_time(self, mask_type: str, time_alert: np.ndarray, mask: np.ndarray) -> float:
        """Get the evaluation time based on the mask type."""
        if mask_type == 'random':
            max_time = (time_alert * mask).max()
            init_time = self.per_init_time * max_time
            return init_time + (max_time - init_time) * np.random.uniform(0, 1)
        elif mask_type == 'three':
            return np.random.choice([16 if self.dataset == "ztf" else 8, 32, 64, 128, 256, 512, 1024, 2048])
        return self.max_time

    def _create_dummy_data(self):
        """Create dummy data for empty cases."""
        return {
            'data': np.zeros((1,), dtype=np.float32),
            'time': np.zeros((1,), dtype=np.float32),
            'mask': np.zeros((1,), dtype=np.float32),
            'var': np.ones((1,), dtype=np.float32),
            'bands': np.zeros((1,), dtype=np.int64),
            'chunk': np.zeros((1,), dtype=np.int64),
            'photometry_mask': np.zeros((1,), dtype=np.float32)
        }

    def _adjust_sequence_length(self, ts_data):
        """Adjust sequence length by truncating or padding."""
        # Handle sampling if not using full light curve
        if self.use_full_lc and self.max_len is not None:
            if len(ts_data['data']) > self.max_len:
                ts_data = self._truncate_sequence(ts_data, self.max_len)
            if len(ts_data['data']) < self.max_len:
                ts_data = self._pad_sequence(ts_data, self.max_len)
        
        if (not self.use_full_lc) and self.max_samples is not None and self.set_type != 'test':
            ts_data = self._random_sample_points(ts_data, self.max_samples)
            if len(ts_data['data']) > self.max_samples:
                ts_data = self._truncate_sequence(ts_data, self.max_samples)
            if len(ts_data['data']) < self.max_samples:
                ts_data = self._pad_sequence(ts_data, self.max_samples)

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
            'var': 1,
            'bands': 0,
            'chunk': -1,  # -1 indicates padding for chunks
            'photometry_mask': 0 # Pad photometry mask with 0
        }
        
        for key in ts_data:
            pad_value = padding_values.get(key, 0)
            result[key] = np.pad(ts_data[key], ((0, padding_len),), 'constant', constant_values=pad_value)
        
        return result

    def _random_sample_points(self, ts_data, n_samples):
        """Randomly sample n_samples points from the time series."""
        if len(ts_data['data']) <= n_samples:
            return ts_data
            
        # Sample indices without replacement
        sample_indices = np.random.choice(len(ts_data['data']), n_samples, replace=False)
        
        # Create new data dict with sampled points
        result = {}
        for key in ts_data:
            result[key] = ts_data[key][sample_indices]
        
        return result

    def _create_sample_dict(self, ts_data, oid_idx):
        """Create the sample dictionary with all fields."""
        sample = {
            "data": ts_data['data'],  # Renamed 'data' to 'magnitude' for augmentation compatibility
            "time": ts_data['time'],
            "mask": ts_data['mask'],
            "bands": ts_data['bands'],
            "var": ts_data['var'],  # Renamed 'var' to 'magnitude_var' for augmentation compatibility
            "chunk": ts_data['chunk'],
            "series_mask": 1 - ts_data['mask'],  # Added series_mask (1 for padding, 0 for valid points)
            "photometry_mask": 1 - ts_data['photometry_mask']
        }
        
        # Get metadata if needed
        if self.use_metadata:
            sample["metadata"] = self.df['metadata'][oid_idx]
        
        # Get features if needed
        if self.use_features:
            sample["features"] = self.df['features'][oid_idx]
        
        return sample

    def _apply_training_masks(self, sample, oid_idx):
        """Apply training masks to the sample."""
        if self.force_online_opt:
            sample = self.mask_data(sample, 'random', oid_idx)
        if self.online_opt_tt:
            sample = self.mask_data(sample, 'three', oid_idx)
        return sample

    def _convert_to_tensors(self, sample):
        """Convert all numpy arrays in sample to PyTorch tensors."""
        result = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value).float()
            else:
                result[key] = value
        return result

    def collate_fn(self, sample: dict, labels: torch.Tensor) -> tuple:
        """Custom collate function for the single-sample data preparation."""
        device = sample["data"].device
        
        # Handle single sample case
        if len(sample["data"].shape) == 1:
            L = sample["data"].shape[0]
            B = 1
        else:
            B, L = sample["data"].shape
            
        # Get tensors for time series (already sorted by time in __getitem__)
        magnitude = sample["data"].float()
        time = sample["time"].float()
        mask = sample["mask"].float()
        bands = sample["bands"].float()
        magnitude_var = sample["var"].float()
        chunk = sample["chunk"].long()
        photometry_mask = sample["photometry_mask"].float()
        # Invert mask to match the expected format (1 for masked, 0 for visible)
        series_mask = 1 - mask
        
        # Create tabular tensor based on whether metadata/features are used
        if self.use_metadata or self.use_features:
            tabular_parts = []
            if self.use_metadata:
                tabular_parts.append(sample["metadata"].float())
            if self.use_features:
                tabular_parts.append(sample["features"].float())
            tabular = torch.cat(tabular_parts, dim=0)
        else:
            # If neither metadata nor features are used, create empty tensor
            tabular = torch.zeros((1), device=device).float()
            
        # Create a mask for tabular data (all visible initially)
        tabular_mask = torch.zeros_like(tabular, device=device)
        
        # Add little noise to tabular data
        if self.set_type == 'train':
            tabular = tabular + torch.rand_like(tabular)*0.01
        
        # Create the final sample dictionary with the expected format
        result = {
            "tabular": tabular,
            "data": magnitude,
            "time": time,
            "bands": bands,
            "tabular_mask": tabular_mask,
            "series_mask": series_mask,
            "var": magnitude_var,
            "chunk": chunk,
            "photometry_mask": photometry_mask,
        }
        
        return result, labels
    
    @property
    def weights(self) -> torch.Tensor:
        """Return the weights calculated during initialization."""
        return self._weights
    
    @weights.setter
    def weights(self, value: torch.Tensor):
        """Set weights."""
        self._weights = value
    
    def get_labels(self) -> np.ndarray:
        """Get the labels for the dataset OIDs."""
        return self.oid_labels.astype(int)
    
    def get_class_names(self) -> List[str]:
        """Get the class names for the dataset."""
        return self.class_names
    
    def get_class_ordered(self) -> List[str]:
        """Get the class names in the preferred order for plotting."""
        return self.class_ordered
    
    def get_class_hierarchy(self) -> Dict[str, List[str]]:
        """Get the class hierarchy for the dataset."""
        return self.class_hierarchy

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Create a dataset from keyword arguments."""
        field_names = {f.name for f in fields(cls)}
        class_kwargs = {k: kwargs[k] for k in kwargs if k in field_names}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in field_names}
        return cls(**class_kwargs, kwargs=extra_kwargs)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, set_type: str, indices: list):
        """Create a dataset from a checkpoint."""
        config_path = os.path.join(checkpoint_path, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls.from_kwargs(**config['data'], set_type=set_type, indices=indices)
