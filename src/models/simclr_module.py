from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from lightly.loss import NTXentLoss
class SimCLRLitModule(LightningModule):
    """SimCLR Lightning Module for self-supervised learning.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: torch.nn.Module,
        projector: torch.nn.Module,
        temperature: float,
        compile: bool,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_epochs: int = 0,
        optimizer: torch.optim.Optimizer = None,

    ) -> None:
        """Initialize a SimCLR Lightning Module.

        :param network: The backbone network to train.
        :param projector: The projection head.
        :param temperature: Temperature parameter for contrastive loss.
        :param compile: Whether to compile the model.
        :param scheduler: The learning rate scheduler to use for training.
        :param warmup_epochs: Number of warmup epochs.
        :param optimizer: The optimizer to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["network", "projector"])

        self.network = network
        self.proyector = projector
        self.loss = NTXentLoss(temperature=temperature)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: A tensor of input data.
        :return: A tensor of projected representations.
        """
        x = self.network(**x)
        return self.proyector(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing two augmented views.

        :return: SimCLR contrastive loss.
        """
        x, y = batch
        z_i = self.forward(x)
        z_j = self.forward(y)
        #
        #batch_size = z_i.shape[0]
        #temperature = self.hparams.temperature
        #
        ## Normalize embeddings
        #z_i = F.normalize(z_i, p=2, dim=1)
        #z_j = F.normalize(z_j, p=2, dim=1)
#
        ## Calculate similarity matrix
        #representations = torch.cat([z_i, z_j], dim=0)
        #similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), #representations.unsqueeze(0), dim=2)
#
        ## Extract positive pairs
        #sim_ij = torch.diag(similarity_matrix, batch_size)
        #sim_ji = torch.diag(similarity_matrix, -batch_size)
        #positives = torch.cat([sim_ij, sim_ji], dim=0)
#
        ## Create mask for valid negative pairs
        #mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(z_j.#device)
#
        ## Calculate loss
        #nominator = torch.exp(positives / temperature)
        #denominator = mask * torch.exp(similarity_matrix / temperature)
        #loss = -torch.log(nominator / torch.sum(denominator, dim=1)).sum() / (2 * #batch_size)
        loss = self.loss(z_i, z_j)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing two augmented views.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)
        batch_size = batch[0]["data"].shape[0]

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing two augmented views.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)
        batch_size = batch[0]["data"].shape[0]

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor and target labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        :return: A tuple containing (representations, labels).
        """
        try:
            x, z, y = batch
        except:
            x, y = batch
        x = self.network(**x)
        return x, y

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()
            self.num_training_batches = len(train_loader)
            
        if self.hparams.compile and (stage == "fit" or stage == "predict"):
            self.network = torch.compile(self.network)
            self.proyector = torch.compile(self.proyector)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.parameters())
        num_training_batches = self.num_training_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_epochs * num_training_batches,
            num_training_steps=self.trainer.max_epochs * num_training_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,      # Every step
                'name': 'cosine_schedule',
            },  
        }


if __name__ == "__main__":
    _ = SimCLRLitModule(None, None, None, None)
