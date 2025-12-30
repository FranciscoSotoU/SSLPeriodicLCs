import h5py
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
import logging
import sys
import os

def dataset_split_handler(set_type, split="0", reduced=False):
    if set_type == "test":
        return "test"
    elif set_type == "validation":
        return f"validation_{split}"

    elif set_type == "training":
        return f"training_{split}"

class ForcedPhotometryDataset(Dataset):
    def __init__(
        self,
        on_evaluation,
        data_dir="",
        data_name="",
        set_type="training",
        split="0",
        max_length=150,
        folded=False,
        random_mask=True,
        random_time=False,
        use_metadata=False,
        use_reduced=False,
        **kwargs,
    ):
        """loading dataset from H5 file"""
        """ dataset is composed for all samples, where self.these_idx dart to sampels for each partition"""

        """ data augmentation methods """
        #read h5 file
        data_path = os.path.join(data_dir, data_name)
        h5_ = h5py.File(data_path, "r")
        set_to_choose = dataset_split_handler(set_type = set_type, split = split, reduced = use_reduced)
        
        self.these_idx = h5_.get(set_to_choose)[:]

        logging.info(f"number of samples = {len(self.these_idx)}")
        self.data = h5_.get("mag_tot")
        self.time = h5_.get("time")
        self.mask = h5_.get("mask")
        self.target = h5_.get("labels") if not use_reduced else h5_.get("labels_reduced")
        self.period = h5_.get("Multiband_period")
        if h5_.get("oid"):
            self.oid = h5_.get("oid")
        else:
            self.oid = None
        self.labels = torch.from_numpy(self.target[:][self.these_idx])
        self.max_lenght = max_length
        self.set_type = set_type
        self.random_mask = random_mask
        self.random_time = random_time
        self.on_evaluation = on_evaluation
        self.folded = folded
        self.use_metadata = use_metadata

    """ obtain validad mask for evaluation prupouses"""

    def random_indexes(self, idx_):
        mask = self.mask[idx_, :, :]
        time = self.time[idx_, :, :]
        data = self.data[idx_, :, :]

        mask_ = []
        time_ = []
        data_ = []

        num_bands = data.shape[1]
        max_length = self.max_lenght

        for i in range(num_bands):
            avv_len = len(mask[:, i])
            available = np.arange(avv_len)[mask[:, i] > 0]
            n_available = np.arange(avv_len)[mask[:, i] < 1]

            max_available = len(available)

            if max_available < max_length:
                if max_available > 0:
                    try:
                        sub_len = random.sample(range(1, max_available), 1)[0]
                        idx_on = random.sample(available.tolist(), sub_len)
                        idx_off = n_available[: max_length - sub_len].tolist()
                        on_off = np.concatenate((idx_on, idx_off))

                        index_diff = max_length - len(on_off)

                        if index_diff > 0:
                            if len(idx_off) > 1:
                                add_idx = np.array(idx_off)[0:1].repeat(int(index_diff))
                                on_off = np.concatenate((on_off, add_idx))
                            else:
                                on_off = np.array(
                                    random.sample(available.tolist(), max_length)
                                )
                        on_off = on_off[np.argsort(on_off)]

                        mask_.append(mask[on_off, i : i + 1])
                        data_.append(data[on_off, i : i + 1])
                        time_.append(time[on_off, i : i + 1])

                    except:
                        mask_.append(mask[:max_length, i : i + 1])
                        data_.append(data[:max_length, i : i + 1])
                        time_.append(time[:max_length, i : i + 1])
                else:
                    mask_.append(mask[:max_length, i : i + 1])
                    data_.append(data[:max_length, i : i + 1])
                    time_.append(time[:max_length, i : i + 1])

            else:
                sub_len = random.sample(range(6, max_length), 1)[0]
                idx_on = random.sample(available.tolist(), sub_len)
                idx_off = n_available[: max_length - sub_len].tolist()
                on_off = np.concatenate((idx_on, idx_off))

                index_diff = max_length - len(on_off)

                if index_diff > 0:
                    if len(idx_off) > 1:
                        add_idx = np.array(idx_off)[0:1].repeat(int(index_diff))
                        on_off = np.concatenate((on_off, add_idx))
                    else:
                        on_off = np.array(random.sample(available.tolist(), max_length))
                on_off = on_off[np.argsort(on_off)]

                mask_.append(mask[on_off, i : i + 1])
                data_.append(data[on_off, i : i + 1])
                time_.append(time[on_off, i : i + 1])

        return (
            np.concatenate(data_, axis=1),
            np.concatenate(time_, axis=1),
            np.concatenate(mask_, axis=1),
        )

    def random_sampling(self, idx_):
        data_dict = {}

        data, time, mask = self.random_indexes(idx_)
        period = self.period[idx_]
        num_bands = data.shape[1]

        mu = []
        st = []
        am = []
        min_time = []

        # data_dict.update({"data": torch.from_numpy(data)})
        # data_dict.update({"time": torch.from_numpy(time)})
        data_dict.update({"mask": torch.from_numpy(mask).float()})
        data_dict.update({"label": torch.from_numpy(np.array(self.target[idx_])).float()})
        data_dict.update({"p": torch.from_numpy(np.array(period)).float()})
        data_dict.update({"idx": idx_})

        """update real time"""
        for i in range(num_bands):
            data_masked = data[mask[:, i] > 0, i]
            if len(data_masked) > 0:
                min_time.append(time[mask[:, i] > 0, i].min())
            else:
                min_time.append(99999999)
        min_time_ = np.min(min_time)
        data_dict.update({"time": torch.from_numpy(time - min_time_)})
        # data_dict.update({"data": torch.from_numpy(data)})

        """update random mask"""
        if self.set_type == "training" and self.random_mask:
            random_time = random.sample([8, 16, 32, 64, 124, 512, 1024, 2048], 1)
            data_dict.update({"mask": data_dict["mask"] * (time <= random_time)})

        mask = data_dict["mask"]
        time = data_dict["time"]

        for i in range(num_bands):
            data_masked = data[mask[:, i] > 0, i]

            if len(data_masked) > 0:
                mu.append(data_masked.mean())
                st.append(data_masked.std())

                """ added """
                span_min = mu[i] - data_masked.min()
                span_max = data_masked.max() - mu[i]
                span = data_masked.max() - data_masked.min()

                am += [span_min, span_max, span]

                data_masked = (data_masked - mu[i]) / (st[i] + 0.001)
                data[mask[:, i] > 0, i] = data_masked

            else:
                mu.append(0)
                st.append(0)
                am += [0, 0, 0]

            # data_.append(data_masked.reshape(-1, 1))
        """minimo de minimos"""

        data_dict.update({"data": torch.from_numpy(data).float()})
        data_dict.update({"mu": torch.from_numpy(np.array(mu)).float()})
        data_dict.update({"st": torch.from_numpy(np.array(st)).float()})
        data_dict.update({"am": torch.from_numpy(np.array(am)).float()})

        data_dict.update(
            {
                "metadata": torch.cat(
                    [
                        data_dict["mu"],
                        data_dict["st"],
                        data_dict["am"],
                    ]
                )
            }
        )

        # if self.set_type == "training" and self.random_time:
        # add random jumps in time
        # delta = (torch.rand(1) * (500 - 0.001) + 1).float()
        # data_dict.update({"time": data_dict["time"]})

        return data_dict
    
    def order_sampling(self, idx_, at_time=None):
        data_dict = {}

        mask = self.mask[idx_, : self.max_lenght, :]
        time = self.time[idx_, : self.max_lenght, :]
        data = self.data[idx_, : self.max_lenght, :]

        period = self.period[idx_]
        num_bands = data.shape[1]
        mu = []
        st = []
        am = []

        # data_dict.update({"data": torch.from_numpy(data)})
        data_dict.update({"time": torch.from_numpy(time).float()})

        #if at_time is not None:
        #    data_dict.update({"mask": torch.from_numpy(mask * (time <= at_time)).float()})
        #else:
        data_dict.update({"mask": torch.from_numpy(mask).float()})

        data_dict.update({"label": torch.from_numpy(np.array(self.target[idx_])).float()})
        data_dict.update({"p": torch.from_numpy(np.array(period)).float()})
        data_dict.update({"idx": idx_})

        for i in range(num_bands):
            data_masked = data[mask[:, i] > 0, i]

            if len(data_masked) > 0:
                mu.append(data_masked.mean())
                st.append(data_masked.std())

                """ added """
                span_min = mu[i] - data_masked.min()
                span_max = data_masked.max() - mu[i]
                span = data_masked.max() - data_masked.min()
                am += [span_min, span_max, span]

                data_masked = (data_masked - mu[i]) / (st[i] + 0.001)
                data[mask[:, i] > 0, i] = data_masked
            else:
                mu.append(0)
                st.append(0)
                am += [0, 0, 0]
            # data_.append(data_masked.reshape(-1, 1))

        data_dict.update({"data": torch.from_numpy(data).float()})
        data_dict.update({"mu": torch.from_numpy(np.array(mu)).float()})
        data_dict.update({"st": torch.from_numpy(np.array(st)).float()})
        data_dict.update({"am": torch.from_numpy(np.array(am)).float()})

        data_dict.update(
            {
                "metadata": torch.cat(
                    [
                        data_dict["mu"],
                        data_dict["st"],
                        data_dict["am"],
                    ]
                )
            }
        )

        return data_dict
    
    
    
    def __getitem__(self, idx, at_det=None):
        """idx is used for pytorch to select samples to construc its batch"""
        """ idx_ is to map a valid index over all samples in dataset  """

        idx_ = self.these_idx[idx]

        """ using or not metadata, in case of using metadata data_dict is updated with new keys """
        """ if folded light curve then time doman is update to a phase"""
        if self.on_evaluation:
            data_dict = self.order_sampling(idx_=idx_)
        else:
            data_dict = self.random_sampling(idx_)
            
        
        # data is data dict without label and metadata
        y = data_dict["label"]
        if self.oid is not None:
            data_dict['oid'] = self.oid[idx_]
        data_dict.pop('label', None)
        return data_dict,y
    
    def __len__(self):
        """lenght of the dataaset, is necesary for consistent getitem values"""
        return len(self.labels)
    #create function for torch balance sampler

    
    def get_labels(self):
        """return labels"""
        return self.labels
    

