from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

class VICRegLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

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
        projector: torch.optim.Optimizer,
        inv_loss_weight: float,
        std_loss_weight: float,
        cov_loss_weight: float,
        compile: bool,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_epochs: int = 0,
        optimizer: torch.optim.Optimizer = None,

    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["encoder",'projector','network'])

        self.network = network
        self.proyector = projector


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        x, y = batch
        x = self.forward(x)
        y = self.forward(y)
        repr_loss = F.mse_loss(x, y)
        N, D = x.shape 
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (N - 1)
        cov_y = (y.T @ y) / (N - 1)
        cov_loss = (off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D))
        return repr_loss, std_loss, cov_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        inv_loss, std_loss,cov_loss = self.model_step(batch)
        batch_size = batch[0]["data"].shape[0]
        weighted_inv_loss = inv_loss * torch.tensor(self.hparams.inv_loss_weight)
        weighted_std_loss = std_loss * torch.tensor(self.hparams.std_loss_weight)
        weighted_cov_loss = cov_loss * torch.tensor(self.hparams.cov_loss_weight)
        loss = weighted_inv_loss + weighted_std_loss + weighted_cov_loss

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("train/inv_loss", inv_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("train/std_loss", std_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("train/cov_loss", cov_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("train/weighted_inv_loss", weighted_inv_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("train/weighted_std_loss", weighted_std_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("train/weighted_cov_loss", weighted_cov_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)

        # return loss or backpropagation will fail
        #print loss if it has a gradient
        return loss


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        inv_loss, std_loss,cov_loss = self.model_step(batch)
        batch_size = batch[0]["data"].shape[0]
        weighted_inv_loss = inv_loss * self.hparams.inv_loss_weight
        weighted_std_loss = std_loss * self.hparams.std_loss_weight
        weighted_cov_loss = cov_loss * self.hparams.cov_loss_weight
        loss = weighted_inv_loss + weighted_std_loss + weighted_cov_loss
        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("val/inv_loss", inv_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("val/std_loss", std_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("val/cov_loss", cov_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("val/weighted_inv_loss", weighted_inv_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("val/weighted_std_loss", weighted_std_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)
        self.log("val/weighted_cov_loss", weighted_cov_loss, on_step=False, on_epoch=True, prog_bar=True,batch_size=batch_size)


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
        # Create parameter groups to avoid weight decay on bias and layer norm weights

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
        
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == "__main__":
    _ = VICRegLitModule(None, None, None,None)
