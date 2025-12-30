from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule

from src.data.components.augmentations.timeseries import *

from src.data.components.datasets.ForcedPhotometryDatasetUnsupervised import ForcedPhotometryDatasetUnsupervised
from torch.utils.data import DataLoader
from src.data.components.collate_functions import collate_dual_dict_trim, collate_dino


class FPUSDatamodule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 0,
        split: int = 0,
        data_name: str = "data",
        transform = None,
        transform_prime = None,
        **kwargs,
    ) -> None:
        """Initialize a `FPV2DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `128`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param split: The split index. Defaults to `0`.
        :param data_name: The name of the dataset. Defaults to `"data"`.
        :param max_length: The maximum sequence length. Defaults to `200`.
        :param balance_sampler: Whether to use an imbalanced dataset sampler. Defaults to `False`.
        :param use_collate: Whether to use a custom collate function. Defaults to `False`.
        :param test_set_type: The type of test set. Defaults to `"test"`.
        :param max_time_to_eval: The maximum time to evaluate. Defaults to `None`.
        """
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.transform_module = TrainTransform(
            transform=transform,
            transform_prime=transform_prime
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if stage == "fit" or stage is None:
            
            self.data_train = ForcedPhotometryDatasetUnsupervised(
            set_type="training",
            transform_module=self.transform_module,
            **self.hparams
            )
            self.data_val = ForcedPhotometryDatasetUnsupervised(
            set_type="validation",
            transform_module=self.transform_module,
            **self.hparams
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle= True,
            collate_fn=collate_dual_dict_trim if not self.hparams.dino else collate_dino,
            persistent_workers= True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_dual_dict_trim if not self.hparams.dino else collate_dino,
            persistent_workers= True if self.hparams.num_workers > 0 else False,
        )


