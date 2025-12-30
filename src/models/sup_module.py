from typing import Any, Dict, Tuple
import warnings

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import torchmetrics
from transformers import get_cosine_schedule_with_warmup
import os
import glob
import seaborn as sns
import numpy as np

# Suppress specific torchmetrics warnings about NaN values in confusion matrix
warnings.filterwarnings("ignore", ".*NaN values found in.*confusion matrix.*", UserWarning)

class SupLitModule(LightningModule):
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
        net: torch.nn.Module,
        classifier: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        num_classes: int = 21,
        freeze_encoder: bool = False,
        pre_trained_path: str = None,
        pre_trained_iteration: int = None,
        class_names: list = None,
        class_names_order: list = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net",'classifier'])

        self.net = net
        self.classifier = classifier
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        metrics = torchmetrics.MetricCollection({'f1': MulticlassF1Score(average='macro', num_classes=num_classes)})
        self.cm_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes, normalize=None)
        self.cm_matrix_test = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes, normalize=None)
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        net_out = self.net(**x)
        classifier_out = self.classifier(net_out)
        return classifier_out

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()

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
        x = batch
        y = batch['label']
        y = y.long()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y , x ,logits
    def model_attn_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x = batch

        attn_lc, attn_ft = self.net.get_attention(**x)
        return batch, attn_lc, attn_ft
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, x, logits = self.model_step(batch)
        self.train_loss(loss)
        self.train_metrics(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, x, logits = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_metrics(preds, targets)
        # Update confusion matrix
        self.cm_matrix(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # Compute confusion matrix (raw counts)
        cm = self.cm_matrix.compute()
        
        # Manually normalize the confusion matrix to avoid NaN issues
        # Normalize each row by its sum (true class distribution)
        row_sums = cm.sum(dim=1, keepdim=True)
        # Avoid division by zero by replacing zero sums with 1
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        cm_normalized = cm.float() / row_sums
        
        # Handle any remaining NaN values (shouldn't happen with above fix, but just in case)
        cm_normalized = torch.nan_to_num(cm_normalized, nan=0.0)
        
        # Get current F1 score
        current_f1 = self.val_metrics['val/f1'].compute()
        
        # Create and save confusion matrix plot
        import matplotlib.pyplot as plt
        
        # Create the plot using the provided function style
        plt.figure(figsize=(12, 10))
        cm_np = cm_normalized.cpu().numpy()
        class_names = self.hparams.class_names
        if self.hparams.class_names_order is not None:
            new_order = [class_names.index(name) for name in self.hparams.class_names_order]
            cm_np = cm_np[np.ix_(new_order, new_order)]
            class_names = self.hparams.class_names_order
        
        plt.imshow(cm_np, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(self.hparams.class_names))
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=10, ha='center')
        plt.yticks(tick_marks, class_names, fontsize=10, va='center')

        thresh = cm_np.max() / 2.

        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                texto = '{0:.2f}'.format(cm_np[i, j])
                plt.text(j, i, texto,
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=10,
                         color="white" if cm_np[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predicted label', fontsize=16)
        
        # Title with F1 score
        cm_title = f'Confusion Matrix - Epoch {self.current_epoch}\nF1 Score: {current_f1:.3f}'
        plt.title(cm_title, fontsize=20, pad=20)
        
        # Determine where to save the confusion matrix
        save_dir = os.path.join(self.trainer.default_root_dir, 'cms')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'confusion_matrix_epoch_{self.current_epoch:03d}.png')
    
        # Save the plot
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        self.cm_matrix.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, x, logits = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_metrics(preds, targets)
        self.cm_matrix_test(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # Compute confusion matrix (raw counts)
        cm = self.cm_matrix_test.compute()
        
        # Manually normalize the confusion matrix to avoid NaN issues
        # Normalize each row by its sum (true class distribution)
        row_sums = cm.sum(dim=1, keepdim=True)
        # Avoid division by zero by replacing zero sums with 1
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        cm_normalized = cm.float() / row_sums
        
        # Handle any remaining NaN values (shouldn't happen with above fix, but just in case)
        cm_normalized = torch.nan_to_num(cm_normalized, nan=0.0)
        
        # Get current F1 score
        current_f1 = self.test_metrics['test/f1'].compute()
        
        # Create and save confusion matrix plot
        import matplotlib.pyplot as plt
        
        # Create the plot using the provided function style
        plt.figure(figsize=(12, 10))
        cm_np = cm_normalized.cpu().numpy()
        
        plt.imshow(cm_np, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(self.hparams.class_names))
        plt.xticks(tick_marks, self.hparams.class_names, rotation=45, fontsize=12, ha='center')
        plt.yticks(tick_marks, self.hparams.class_names, fontsize=12, va='center')

        thresh = cm_np.max() / 2.

        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                texto = '{0:.2f}'.format(cm_np[i, j])
                plt.text(j, i, texto,
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=15,
                         color="white" if cm_np[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predicted label', fontsize=16)
        
        # Title with F1 score
        cm_title = f'Confusion Matrix - Test \nF1 Score: {current_f1:.3f}'
        plt.title(cm_title, fontsize=20, pad=20)
        
        # Determine where to save the confusion matrix
        save_dir = os.path.join(self.trainer.default_root_dir, 'cms')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'confusion_matrix_epoch_test.png')
    
        # Save the plot
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        self.cm_matrix.reset()
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """


        idx = batch['idx']
        loader_idx = batch['loader_idx']
        loss, preds, targets, x, logits = self.model_step(batch)
        return preds, targets, idx, logits.float(), x, loader_idx

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        if self.hparams.compile and (stage == "fit" or stage == "predict"):
            self.net = torch.compile(self.net)
            self.classifier = torch.compile(self.classifier)

        if self.hparams.pre_trained_path is not None and stage == "fit":

            pre_trained_path = str(self.hparams.pre_trained_path)
            assert os.path.exists(pre_trained_path), f"Pre-trained weights file not found: {pre_trained_path}"
            if self.hparams.pre_trained_iteration is not None:
                pre_trained_path = os.path.join(self.hparams.pre_trained_path, str(self.hparams.pre_trained_iteration))
    
            ckpt_files = glob.glob(os.path.join(pre_trained_path, "**", "*.ckpt"), recursive=True)

            ckpt_files = [f for f in ckpt_files if 'last' not in f]
            assert len(ckpt_files) > 0, f"No ckpt files found in {pre_trained_path}"
            ckpt_path = ckpt_files[0]
            print(f"Loading pre-trained weights from {ckpt_path}")
            #ckpt_path = pre_trained_path
    

        
            state_dict = torch.load(ckpt_path)['state_dict']
            state_dict = {
                key.replace("network.", ""): value
                for (key, value) in state_dict.items() if key.startswith("network.")
            }
            device = next(self.net.parameters()).device
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
    
            # Load the state dictionary into the model
            self.net.load_state_dict(state_dict)
            print(f"Loaded pre-trained weights")
        
        if self.hparams.freeze_encoder and stage == "fit":
            print("Freezing encoder")
            self.net.requires_grad_(False)
            self.classifier.requires_grad_(True)
            
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # Create parameter groups to avoid weight decay on bias and layer norm weights

        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            #"lr_scheduler": {
            #    "scheduler": scheduler,
            #    "interval": "step",
            #    "frequency": 1,
            #    "reduce_on_plateau": False,
            #},
        }


if __name__ == "__main__":
    _ = SupLitModule(None, None, None,None)