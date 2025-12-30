import numpy as np
import torch
import random


def random_noise(data_dict, **kwargs):
    probability = kwargs.get('probability', 1)
    if probability < np.random.rand():
        return data_dict
    mag = data_dict['data']
    error = data_dict['error']
    std = kwargs.get('std', 1)
    noise_value = torch.normal(mean=0, std=std, size=mag.shape).float()
    mag = mag + noise_value * error
    data_dict.update({"data": mag})
    return data_dict

def random_scaling(data_dict, **kwargs):
    probability = kwargs.get('probability', 1)
    if probability < np.random.rand():
        return data_dict
    mag = data_dict['data']
    scale = kwargs.get('scale', 0.2)  # Default scale of 0.2 gives range of [0.8, 1.2]
    scale_factor = 1.0 + torch.normal(mean=0, std=scale, size=mag.shape).clamp(-0.2, 0.2)
    mag = mag * scale_factor
    data_dict.update({"data": mag})
    return data_dict
    
def time_inverse(data_dict, **kwargs):
    probability = kwargs.get('probability', 1)
    if probability < np.random.rand():
        return data_dict
    mask = data_dict['mask']
    time = data_dict['time']
    mag = data_dict['data']
    error = data_dict['error']
    time = -time
    # Sort time along the 0th dimension
    pad_mask = mask==0
    highest_value = 1e10 * pad_mask.float()
    time = time + highest_value
    time, indices = time.sort(0)
    #time = time - torch.where(temporal_mask, torch.tensor(1e10), torch.tensor(0.0))
    mag = torch.gather(mag, 0, indices)
    error = torch.gather(error, 0, indices)
    mask = torch.gather(mask, 0, indices)
    #time = time - torch.tensor(1e10)
    time =  (time - time.min()) * mask
   # print(time.shape)
    data_dict.update({"mask": mask})
    data_dict.update({"time": time})
    data_dict.update({"data": mag})
    data_dict.update({"error": error})

    return data_dict


def amplitude_inverse(data_dict, **kwargs):
    probability = kwargs.get('probability', 1)
    if probability < np.random.rand():
        return data_dict
    mask = data_dict['mask']
    mag = data_dict['data']
    mean = mag[mask>0].mean()
    mag = -mag + 2*mean
    data_dict.update({"data": mag}) 
    return data_dict

def random_mask(data_dict,**kwargs):
    probability = kwargs.get('probability', 1)
    if probability < np.random.rand():
        return data_dict
    mask = data_dict['mask']
    data = data_dict['data']
    error = data_dict['error']
    time = data_dict['time']
    if  mask.sum() > 12 :
        max_points = np.random.randint(6, mask.sum().item())
        mask = torch.clone(mask)
        mask[max_points:,:] = 0

        data_dict.update({"mask": mask})
        data_dict.update({"data": data*mask.float()})
        data_dict.update({"error": error*mask.float()})
        data_dict.update({"time": time*mask.float()})
    return data_dict

def random_mask_time(data_dict, **kwargs):
    probability = kwargs.get('probability', 0)
    if probability > np.random.rand():
        return data_dict
    time = data_dict['time']
    mask = data_dict['mask']
    random_time = random.sample([8, 16, 32, 64, 124, 512, 1024, 2048,4096], 1)[0]
    filter_time = time <= random_time
    mask = mask * filter_time
    data_dict.update({"mask": mask})
    return data_dict

def time_shift(data_dict, **kwargs):
    probability = kwargs.get('probability', 1)
    if probability < np.random.rand():
        return data_dict
    time = data_dict['time']
    mask = data_dict['mask']
    mag = data_dict['data']
    error = data_dict['error']
    time_flatten = time[mask>0].flatten()
    first_time_window = torch.max(time_flatten) - torch.min(time_flatten)
    random_time = np.random.choice(time_flatten)
    time = time - random_time
    temporal_mask = time < 0 
    
    negative_time = time * temporal_mask.float() * mask.float() - time.min()
    negative_time = negative_time + random_time
    second_time_window = torch.max(negative_time)
    diff = first_time_window - second_time_window
    negative_time = negative_time + diff
    time[temporal_mask] = negative_time[temporal_mask]
    pad_mask = mask==0
    highest_value = 1e10 * pad_mask.float()
    time = time + highest_value

    time, indices = time.sort(0)
    mag = torch.gather(mag, 0, indices)
    error = torch.gather(error, 0, indices)
    mask = torch.gather(mask, 0, indices)
    highest_value = torch.gather(highest_value, 0, indices)
    time = time - highest_value
    data_dict.update({"time": time})
    data_dict.update({"data": mag})
    data_dict.update({"error": error})
    data_dict.update({"mask": mask})
    return data_dict

def constant_time_warping(data_dict, **kwargs):
    probability = kwargs.get('probability', 1)
    scale = kwargs.get('scale', 0.05)
    scale =  torch.empty(1).uniform_(1-scale, 1+scale).item()
    if probability < np.random.rand():
        return data_dict
    time = data_dict['time']
    period = data_dict['p']
    time = time * (1 + scale)
    data_dict.update({"time": time})
    period = period * (1 + scale)
    data_dict.update({"p": period})


    return data_dict
