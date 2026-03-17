from scipy.interpolate import CubicSpline
import numpy as np
from random import random
import torch
import copy
def apply_transform(prob):
    return random() <= prob


class DINOAugmentation(object):
    """
    DINO-style augmentation that generates 10 views: 2 global and 8 local crops.
    Adapted for time series lightcurve data.
    
    Parameters:
    - global_crop_scale: (min, max) fraction of original sequence for global views
    - local_crop_scale: (min, max) fraction of original sequence for local views  
    - num_local_crops: number of local views to generate (default 8)
    - max_length: maximum length to pad crops to
    """
    def __init__(self, 
                 global_crop_scale=(0.4, 1.0),    # Global views: 40-100% of sequence
                 local_crop_scale=(0.05, 0.4),    # Local views: 5-40% of sequence
                 num_local_crops=8,
                 dino_prob=1.0,
                 max_length=512,
                 **kwargs):
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.num_local_crops = num_local_crops
        self.prob = dino_prob
        self.max_length = max_length
        
        print(f'Initializing DINO Augmentation:')
        print(f'  - Global crop scale: {self.global_crop_scale} (fraction of sequence)')
        print(f'  - Local crop scale: {self.local_crop_scale} (fraction of sequence)')
        print(f'  - Number of local crops: {self.num_local_crops}')
        print(f'  - Probability: {self.prob}')
        print(f'  - Max length: {self.max_length}')

    def get_crop_params(self, seq_len, scale_range):
        """
        Get random crop parameters for time series.
        
        Args:
            seq_len: Length of original sequence
            scale_range: (min_scale, max_scale) - fraction of original sequence
            
        Returns:
            start_idx: Starting index for crop
            crop_len: Length of crop
        """
        # Random scale (fraction of original sequence length)
        scale = np.random.uniform(scale_range[0], scale_range[1])
        crop_len = int(seq_len * scale)
        crop_len = max(1, min(crop_len, seq_len))  # Ensure valid crop length
        
        # Random starting position
        max_start = seq_len - crop_len
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        return start_idx, crop_len, scale

    def apply_crop(self, data_dict, start_idx, crop_len, scale):
        """Apply temporal cropping and padding to all data components."""
        cropped_dict = {}
        target_length = int(self.max_length * scale)
        
        for key in ['data', 'time', 'mask', 'bands', 'error']:
            if key in data_dict:
                original_data = data_dict[key]
                if len(original_data) > 0:
                    end_idx = min(start_idx + crop_len, len(original_data))
                    cropped_data = original_data[start_idx:end_idx]
                    
                    # For time key, subtract the minimum value to start from 0
                    if key == 'time' and len(cropped_data) > 0:
                        cropped_data = cropped_data - np.min(cropped_data)
                    
                    # Pad to target length
                    if len(cropped_data) < target_length:

                        pad_size = target_length - len(cropped_data)
                        padding = np.zeros(pad_size, dtype=cropped_data.dtype)

                        cropped_data = np.concatenate([cropped_data, padding])
                    
                    cropped_dict[key] = cropped_data
                else:
                    cropped_dict[key] = original_data
        
        return cropped_dict

    def __call__(self, data_dict):
        # Get sequence length
        seq_len = len(data_dict['data'])
        if seq_len == 0:
            return data_dict
        
        views = []
        
        # Generate 2 global views
        for i in range(2):
            start_idx, crop_len, scale = self.get_crop_params(seq_len, self.global_crop_scale)
            global_view = self.apply_crop(data_dict, start_idx, crop_len, scale)
            views.append(global_view)
        
        # Generate 8 local views
        for i in range(self.num_local_crops):
            start_idx, crop_len, scale = self.get_crop_params(seq_len, self.local_crop_scale)
            local_view = self.apply_crop(data_dict, start_idx, crop_len, scale)
            views.append(local_view)
        
        # Return dictionary with all views
        result = {
            'views': views
        }
        
        return result

    def __str__(self):
        return "DINO Augmentation"



class SequentialCompose(object):
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms

    def __call__(self, data_dict):
        if len(self.transforms) == 0:
            return data_dict
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
    
class RandomCompose(object):
    def __init__(self, transforms, weights=None, **kwargs):
        self.transforms = transforms
        if weights is not None:
            total = sum(weights)
            self.probs = [w / total for w in weights]
        else:
            self.probs = None

    def __call__(self, data_dict):
        if not self.transforms:
            return data_dict
        indices = np.arange(len(self.transforms))
        if self.probs is not None:
            # For each transform, decide to apply based on its probability
            for i, t in enumerate(self.transforms):
                if np.random.rand() < self.probs[i]:
                    data_dict = t(data_dict)
        else:
            # If no weights, apply all transforms in random order
            np.random.shuffle(indices)
            for i in indices:
                data_dict = self.transforms[i](data_dict)
        return data_dict



class Scaling(object):
    def __init__(self, scaling_sigma_min=0.1,scaling_sigma_max=10., scaling_prob=0.7, **kwargs):
        self.sigma = (scaling_sigma_min, scaling_sigma_max)
        self.prob = scaling_prob
        print('Initializing Scaling with sigma:', self.sigma, 'and probability:', self.prob)

    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        mag = data_dict['data']
        #scaler is a value between max and min sigma with shape 1
        scaler = np.random.uniform(self.sigma[0], self.sigma[1])
        mag = mag * scaler
        data_dict.update({"data": mag.astype(np.float32)})
        return data_dict
    def __str__(self):
        return "Scaling"


class FlipTimeSignal(object):
    def __init__(self, flip_time_signal_prob=0.3, **kwargs):
        self.prob =  float(flip_time_signal_prob)
        print('Initializing FlipTimeSignal with probability:', self.prob)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        time = data_dict['time']
        mask = data_dict['mask']
        mag = data_dict['data']
        bands = data_dict['bands']

        error = data_dict['error']
        time = np.abs(time - np.max(time))
        time = np.flip(time)
        mag = np.flip(mag)
        bands = np.flip(bands)
        mask = np.flip(mask)
        error = np.flip(error)
        data_dict['error'] = error
        data_dict['time'] = time
        data_dict['data'] = mag
        data_dict['mask'] = mask
        data_dict['bands'] = bands
        return data_dict
    def __str__(self):
        return "Flip Time Signal"


class TimeShift(object):
    def __init__(self, time_shift_prob=0.8,time_shift_sigma=0.2, **kwargs):
        self.prob = time_shift_prob
        self.sigma = time_shift_sigma
        print('Initializing TimeShift with probability:', self.prob, 'and sigma:', self.sigma)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
            
        time = data_dict['time']
        mask = data_dict['mask']
        mag = data_dict['data']
        bands = data_dict['bands']
        error = data_dict['error']
        
        # Get the current time span
        time_span = np.max(time) - np.min(time)
        # Choose a random split point (but not at the very beginning or end)
        n_points = mask.sum()
        if n_points < 10:
            return data_dict
            
        # Choose split point between 10% and 90% of the data
        max_points = n_points * self.sigma
        
        split_idx = np.random.randint(0, max_points)
        if split_idx < 1:
            return data_dict
        # Split the time series
        time_part1 = time[:split_idx]
        time_part2 = time[split_idx:]
        
        # Shift part2 to start at time 0, then shift part1 to come after part2
        time_part2_shifted = time_part2 - time_part2[0]
        time_part1_shifted = time_part1 - time_part1[0] + time_part2_shifted[-1] 
        time_span_2 = np.max(time_part1_shifted) - np.min(time_part2_shifted)
        # Combine the shifted parts
        diff_span = time_span_2 - time_span
        time_part1_shifted += np.abs(diff_span) 
        time_shifted = np.concatenate([time_part2_shifted, time_part1_shifted])

        # Create the corresponding reordering for other arrays
        mag_shifted = np.concatenate([mag[split_idx:], mag[:split_idx]])
        mask_shifted = np.concatenate([mask[split_idx:], mask[:split_idx]])
        bands_shifted = np.concatenate([bands[split_idx:], bands[:split_idx]])
        error_shifted = np.concatenate([error[split_idx:], error[:split_idx]])
        
        data_dict.update({
            "data": mag_shifted,
            "time": time_shifted,
            "mask": mask_shifted,
            "bands": bands_shifted,
            "error": error_shifted
        })
        
        return data_dict
        
    def __str__(self):
        return "Time Shift"

class RandomMask(object):
    def __init__(self, random_mask_min_points=16, random_mask_prob=0.6, **kwargs):
        self.min_points = random_mask_min_points
        self.prob = random_mask_prob
        print('Initializing RandomMask with min points:', self.min_points, 'and probability:', self.prob)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        mask = data_dict['mask']
        length = mask.sum()
        if length > self.min_points:
            keep_ratio = np.random.uniform(0.2, 1)
            max_points = max(self.min_points, int(length * keep_ratio))
            
            # Use slicing instead of copying all arrays first
            data_dict.update({
                "mask": data_dict['mask'][:max_points].astype(np.float32),
                "data": data_dict['data'][:max_points].astype(np.float32),
                "time": data_dict['time'][:max_points].astype(np.float32),
                "bands": data_dict['bands'][:max_points].astype(np.float32),
                "error": data_dict['error'][:max_points]
            })
        return data_dict
    def __str__(self):
        return "Random Mask"

class AmplitudeInverse(object):
    def __init__(self, amplitude_inverse_prob=0.2, **kwargs):
        self.prob = amplitude_inverse_prob
        print('Initializing AmplitudeInverse with probability:', self.prob)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        mag = data_dict['data']
        mean = mag.mean()
        mag = -mag + 2 * mean
        data_dict.update({"data": mag.astype(np.float32)})
        return data_dict
    def __str__(self):
        return "Amplitude Inverse"
    

class Masking(object):
    def __init__(self, masking_prob=0.5, **kwargs):
        self.prob = masking_prob
        print('Initializing Masking with probability:', self.prob)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        mask = data_dict['mask']
        data = data_dict['data']
        time = data_dict['time']
        bands = data_dict['bands']
        error = data_dict['error']
        length = len(mask)
        sigma = np.random.uniform(0.2, 1.0)
        init = 0
        num_to_mask = int(length * sigma)
        if length < 9 or num_to_mask < 1:
            return data_dict
        selected = int(np.random.uniform(init, num_to_mask))
        max_init = max(length - selected, 2)
        init = np.random.randint(init, max_init)
        mask[init:(init + selected)] = 0
        data = data[mask > 0]
        time = time[mask > 0]
        bands = bands[mask > 0]
        error = error[mask > 0]
        data_dict['error'] = error
        mask = mask[mask > 0]
        data_dict.update({
            "mask": mask.astype(np.float32),
            "data": data.astype(np.float32),
            "time": time.astype(np.float32),
            "bands": bands.astype(np.float32)
        })
        return data_dict
    def __str__(self):
        return "Masking"
    
class Jitter(object):
    def __init__(self, jitter_sigma=0.05, jitter_prob=0.9, **kwargs):
        self.sigma, self.prob = jitter_sigma, jitter_prob
        print('Initializing Jitter with sigma:', self.sigma, 'and probability:', self.prob)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        mag = data_dict['data']
        time = data_dict['time']

        mag = mag + np.random.normal(0, self.sigma, mag.shape) 
        time = time + np.random.normal(0, self.sigma, time.shape)

        data_dict.update({
            "data": mag.astype(np.float32),
            "time": time.astype(np.float32)
        })
        return data_dict
    def __str__(self):
        return "Jitter"

class Stretch(object):
    def __init__(self, stretch_sigma=0.05, stretch_prob=0.7, **kwargs):
        self.sigma, self.prob = stretch_sigma, stretch_prob
        print('Initializing Stretch with sigma:', self.sigma, 'and probability:', self.prob)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        time = data_dict['time']
        stretch = np.random.normal(1, self.sigma)
        time = time * stretch
        data_dict.update({
            "time": time.astype(np.float32),
        })
        return data_dict
    def __str__(self):
        return "Stretch"
    
    
    
class RandomNoise(object):
    
    def __init__(self, noise_sigma=1, random_noise_prob=0.5, **kwargs):
        self.sigma, self.prob = noise_sigma, random_noise_prob
        print('Initializing RandomNoise with sigma:', self.sigma, 'and probability:', self.prob)

    def __call__(self, data_dict):
        """_summary_

        Args:
            data_dict (dict): Dictionary containing the all light curve data(time, data, mask, bands)

        Returns:
            dict: Dictionary containing the all light curve data updated with random noise
        """
        if not apply_transform(self.prob):
            return data_dict

        mag = data_dict['data']
        error = data_dict['error']
        noise = mag + np.random.normal(0, self.sigma, mag.shape) * error
        #create a random value between 0 and 1
        mag = mag + noise
        data_dict.update({"data": mag.astype(np.float32)})
        return data_dict
    def __str__(self):
        return "Random Noise"
    
class RandomNoise2(object):
    
    def __init__(self, noise_sigma=1, random_noise_prob=0.5, **kwargs):
        self.sigma, self.prob = noise_sigma, random_noise_prob
        print('Initializing RandomNoise with sigma:', self.sigma, 'and probability:', self.prob)

    def __call__(self, data_dict):
        """_summary_

        Args:
            data_dict (dict): Dictionary containing the all light curve data(time, data, mask, bands)

        Returns:
            dict: Dictionary containing the all light curve data updated with random noise
        """
        if not apply_transform(self.prob):
            return data_dict

        mag = data_dict['data']
        error = data_dict['error']
        noise = np.random.normal(0, self.sigma, mag.shape) * error
        #create a random value between 0 and 1
        mag = mag + noise
        data_dict.update({"data": mag.astype(np.float32)})
        return data_dict
    def __str__(self):
        return "Random Noise"

class SignalResampling(object):
    def __init__(self, signal_resampling_prob=0.5, signal_resampling_sigma=0.8, **kwargs):
        self.prob = signal_resampling_prob
        self.sigma = signal_resampling_sigma
        print('Initializing SignalResampling with probability:', self.prob, 'and sigma:', self.sigma)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        time = data_dict['time']
        mask = data_dict['mask']
        mag = data_dict['data']
        bands = data_dict['bands']
        error = data_dict['error']
        n_points = mask.sum()
        if n_points < 9:
            return data_dict
        sigma = np.random.uniform(0.1, self.sigma)
        n_points_ = int(n_points * sigma)
        if n_points_ < 9:
            return data_dict
        removed_indices = np.random.choice(np.arange(n_points,dtype=int), n_points_, replace=False)
        time = np.delete(time, removed_indices)
        mag = np.delete(mag, removed_indices)
        bands = np.delete(bands, removed_indices)
        mask = np.delete(mask, removed_indices)
        error = np.delete(error, removed_indices)
        data_dict['error'] = error
        data_dict.update({
            "time": time.astype(np.float32),
            "data": mag.astype(np.float32),
            "bands": bands.astype(np.float32),
            "mask": mask.astype(np.float32)
        })
        return data_dict
    def __str__(self):
        return "Signal Resampling"

class ChannelShuffling(object):
    def __init__(self, channel_shuffling_prob=0.3, **kwargs):
        self.prob = channel_shuffling_prob
        print('Initializing ChannelShuffling with probability:', self.prob)
    def __call__(self, data_dict):
        if not apply_transform(self.prob):
            return data_dict
        bands = data_dict['bands']
        bands_ = np.where(bands == 1, 2, np.where(bands == 2, 1, bands))
        data_dict['bands'] = bands_.astype(np.float32)
        return data_dict
    def __str__(self):
        return "Channel Shuffling"


class Identity(object):
    """Identity transformation, does nothing to the data."""
    def __init__(self, **kwargs):
        print('Initializing Identity transformation')
    def __call__(self, data_dict):
        """Returns the input data dictionary unchanged."""
        return data_dict
    
    def __str__(self):
        return "Identity"
    
class ToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data_dict):
        for key in data_dict:
            data_dict[key] = torch.tensor(data_dict[key])
        return data_dict
    
class TrainTransform(object):
    def __init__(self, transform, transform_prime):
        self.transform = transform
        self.transform_prime = transform_prime
        self.dino = False
        
    def __call__(self, data_dict):
        x1 = self.transform(copy.deepcopy(data_dict))
        x2 = self.transform_prime(copy.deepcopy(data_dict))
        return x1, x2

class DinoTrainTransform(object):
    def __init__(self, transform, transform_prime):
        self.transform = transform
        self.transform_prime = transform_prime
        self.dino = True

    def __call__(self, data_dict):
        x1 = self.transform(data_dict)
        return x1

class TransformModule(object):
    def __init__(self, transform_list=[],mode='linear',**kwargs):
        """Set the transform module to be used for data augmentation.

        :param transform_module: The transform module to set.
        """
        aug_functions = {
            'time_inverse': FlipTimeSignal,
            'scaling': Scaling,
            'random_mask': RandomMask,
            'time_shift': TimeShift,
            'amplitude_inverse': AmplitudeInverse,
            'jitter': Jitter,
            'stretch': Stretch,
            'resampling': SignalResampling,
            'channel_shuffle': ChannelShuffling,
            'masking': Masking,
            'random_noise': RandomNoise,
            'dino': DINOAugmentation,
            'random_noise2':RandomNoise2,
        }
        aug_list = []

        for aug_name in transform_list:
            if aug_name in aug_functions and aug_name != 'dino':
                aug_list.append(aug_functions[aug_name](**kwargs))
        aug_list.append(Identity())
        if 'dino' in transform_list:
            self.compose = DinoCompose(aug_list)
        else:
            self.compose = SequentialCompose(aug_list)
    def __call__(self, data_dict):
        return self.compose(data_dict)


class DinoCompose(object):
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms
        self.dino_transform = DINOAugmentation(**kwargs)
    def __call__(self, data_dict):
        views = self.dino_transform(data_dict)
        new_views = []
        for  view in views['views']:
            for transform in self.transforms:
                view = transform(view)
            new_views.append(view)
        new_views = {'views': new_views}
        return new_views
