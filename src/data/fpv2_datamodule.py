from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule

from .components.datasets.ForcedPhotometryDatasetMM import ForcedPhotometryDatasetMM
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from .components.collate_functions import collate_trim_to_max_len, default_collate_dict

class FPV2DataModule(LightningDataModule):
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
        max_length: int = 200,
        balance_sampler: bool = False,
        use_collate: bool = False,
        test_set_type: str = "test",
        max_time_to_eval: int = None,
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


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if stage == "fit" or stage is None:
            self.data_train = ForcedPhotometryDatasetMM(
            set_type="train",
            **self.hparams
            )
            self.data_val = ForcedPhotometryDatasetMM(
            set_type="valid",
            on_evaluation=True,
            **self.hparams
            )
        if stage == "test":
            self.data_test = ForcedPhotometryDatasetMM(
            set_type=self.hparams.test_set_type,
            on_evaluation=True,
            **self.hparams
            )
        if stage == 'predict':
            self.data_test = ForcedPhotometryDatasetMM(
            set_type=self.hparams.test_set_type,
            on_evaluation=True,
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
            shuffle= True if not self.hparams.balance_sampler else False,
            sampler=ImbalancedDatasetSampler(self.data_train) if self.hparams.balance_sampler else None,
            collate_fn=collate_trim_to_max_len if self.hparams.use_collate else None,
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
            collate_fn=collate_trim_to_max_len if self.hparams.use_collate else None,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_trim_to_max_len if self.hparams.use_collate else None,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_trim_to_max_len if self.hparams.use_collate else None,
            
        )
if __name__ == "__main__":
    _ = FPV2DataModule()
