from typing import Any, Dict, Tuple
import warnings

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import torchmetrics
from transformers import get_cosine_schedule_with_warmup
import os
import glob
import seaborn as sns
import numpy as np
from src.models.components.utils.focal_loss import FocalLoss


# Suppress specific torchmetrics warnings about NaN values in confusion matrix
warnings.filterwarnings("ignore", ".*NaN values found in.*confusion matrix.*", UserWarning)

class EFATATModule(LightningModule):
    """Enhanced FATAT Module - simplified with single encoder and classifier.

    A unified architecture using one encoder network and one classifier.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        encoder: torch.nn.Module = None,
        classifier: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        compile: bool = False,
        num_classes: int = 21,
        freeze_encoder: bool = False,
        pre_trained_path: str = None,
        pre_trained_iteration: int = None,
        class_names: list = None,
        class_names_order: list = None,
        criterion: str = 'cross_entropy',
        warmup_epochs: int = 1,
        UMAP: bool = False,
        plot_all_epochs_cms: bool = False,
        dino: bool = False
    ) -> None:
        """Initialize EFATATModule.

        :param encoder: The encoder network.
        :param classifier: The classifier network.
        :param optimizer: The optimizer to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['encoder', 'classifier'])

        self.encoder = encoder
        self.classifier = classifier

        if pre_trained_path is not None and self.encoder is not None:
            self.load_model(pre_trained_path, iteration=pre_trained_iteration)
        if self.encoder is not None and freeze_encoder:
            self.encoder.requires_grad_(False)
        # loss function
        self.criterion = FocalLoss() if criterion == 'focal_loss' else torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        metrics = torchmetrics.MetricCollection({'f1': MulticlassF1Score(average='macro', num_classes=num_classes)})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        # confusion matrix metrics
        self.val_cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes, normalize='true')
        self.test_cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes, normalize="true")
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # get best loss
        self.val_loss_min = MinMetric()
        
        # Statistics tracking for encoder outputs
        self.train_encoder_mean = MeanMetric()
        self.train_encoder_var = MeanMetric()
        self.val_encoder_mean = MeanMetric()
        self.val_encoder_var = MeanMetric()
        
        self.UMAP = UMAP 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: Input batch dictionary.
        :return: A tensor of logits.
        """
        if self.hparams.freeze_encoder and self.encoder is not None:
            self.encoder.eval()
            
        encoder_out = self.encoder(**x)
        logits = self.classifier(encoder_out)
        return logits, encoder_out
    
    def forward_latent(self, x: torch.Tensor):
        """Perform a forward pass and return the latent representations.

        :param x: Input batch dictionary.
        :return: Latent representations from encoder.
        """
        encoder_out = self.encoder(**x)
        return encoder_out
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_cm.reset()
        self.val_loss_min.reset()
        self.val_encoder_mean.reset()
        self.val_encoder_var.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        y = batch['label'].long()
        logits, encoder_out = self.forward(batch)
        
        # Track statistics for encoder output
        encoder_mean = encoder_out.mean()
        encoder_var = encoder_out.var()
        self.train_encoder_mean(encoder_mean)
        self.train_encoder_var(encoder_var)
        
        # Calculate loss
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.train_loss(loss)
        self.train_metrics(preds, y)
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/encoder_mean", self.train_encoder_mean, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/encoder_var", self.train_encoder_var, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
    
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
        y = batch['label'].long()
        logits, encoder_out = self.forward(batch)
        
        # Track statistics for encoder output
        encoder_mean = encoder_out.mean()
        encoder_var = encoder_out.var()
        self.val_encoder_mean(encoder_mean)
        self.val_encoder_var(encoder_var)
        
        # Calculate loss
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_loss(loss)
        self.val_metrics(preds, y)
        self.val_cm(preds, y)
        
        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/encoder_mean", self.val_encoder_mean, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/encoder_var", self.val_encoder_var, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        current_val_loss = self.val_loss.compute()
        self.val_loss_min.update(current_val_loss)
        new_best = self.val_loss_min.compute()
        
        is_best = current_val_loss <= new_best
        
        # Save best confusion matrix
        if is_best:
            self.plot_confusion_matrix(stage='val', epoch=self.current_epoch, best=True)
        
        # Save epoch confusion matrix if plot_all_epochs_cms is True
        if self.hparams.plot_all_epochs_cms:
            self.plot_confusion_matrix(stage='val', epoch=self.current_epoch, best=False)
                    
        # Reset confusion matrix for the next epoch
        self.val_cm.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        y = batch['label'].long()
        logits, encoder_out = self.forward(batch)
        
        # Calculate loss
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.test_loss(loss)
        self.test_metrics(preds, y)
        self.test_cm(preds, y)
        
        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.plot_confusion_matrix(stage='test')

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        oid = batch['oid']
        targets = batch['label'].long()
        
        if self.UMAP:
            # If UMAP is enabled, we only return the latent representations
            latent = self.forward_latent(batch)
            return {
                "latent": latent,
                "oid": oid,
                "targets": targets,
            }
        else:
            logits, encoder_out = self.forward(batch)
  
        return {
            "logits": logits,
            "targets": targets,
            "oid": oid,
        }

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
            if self.encoder is not None:
                self.encoder = torch.compile(self.encoder)
            if self.classifier is not None:
                self.classifier = torch.compile(self.classifier)

            
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
    def plot_confusion_matrix(self, stage='val', epoch=None, best=False) -> str:
        """
        Create and save a confusion matrix plot for the specified stage.
        
        Args:
            stage: Stage name ('val', 'test')
            epoch: Current epoch number (None for test)
            best: Whether this is the best model
        
        Returns:
            str: Path where the confusion matrix was saved
        """
        import matplotlib.pyplot as plt
        
        # Get the appropriate confusion matrix and F1 score
        if stage == 'val':
            cm = self.val_cm.compute()
            current_f1 = self.val_metrics['val/f1'].compute()
        else:  # test
            cm = self.test_cm.compute()
            current_f1 = self.test_metrics['test/f1'].compute()
        
        # Manually normalize the confusion matrix to avoid NaN issues
        cm_normalized = cm.float()
        # Create the plot
        plt.figure(figsize=(12, 10))
        cm_np = cm_normalized.cpu().numpy()
        class_names = self.hparams.class_names
        
        # Reorder classes if specified
        if self.hparams.class_names_order is not None:
            new_order = [class_names.index(name) for name in self.hparams.class_names_order]
            cm_np = cm_np[np.ix_(new_order, new_order)]
            class_names = self.hparams.class_names_order
        
        plt.imshow(cm_np, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        
        # Adjust styling
        rotation = 90
        fontsize = 10 if self.hparams.num_classes > 10 else 12
        text_fontsize = 10 if self.hparams.num_classes > 10 else 12
        
        plt.xticks(tick_marks, class_names, rotation=rotation, fontsize=fontsize, ha='center')
        plt.yticks(tick_marks, class_names, fontsize=fontsize, va='center')

        thresh = cm_np.max() / 2.
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                texto = '{0:.2f}'.format(cm_np[i, j])
                plt.text(j, i, texto,
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=text_fontsize,
                         color="white" if cm_np[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predicted label', fontsize=16)
        
        # Create title and filename based on stage
        stage_str = stage.capitalize()
        if stage == 'test':
            cm_title = f'Confusion Matrix - {stage_str}\nF1 Score: {current_f1:.3f}'
            filename = f'confusion_matrix_{stage}.png'
            save_dir = os.path.join(self.trainer.default_root_dir, 'cms')
        else:
            if best:
                cm_title = f'Confusion Matrix - Best {stage_str}\nF1 Score: {current_f1:.3f}'
                filename = f'confusion_matrix_best_{stage}.png'
                save_dir = os.path.join(self.trainer.default_root_dir, 'cms')
            else:
                cm_title = f'Confusion Matrix - Epoch {epoch} {stage_str}\nF1 Score: {current_f1:.3f}'
                filename = f'confusion_matrix_epoch_{epoch:03d}_{stage}.png'
                save_dir = os.path.join(self.trainer.default_root_dir, 'cms', 'epochs')
        plt.title(cm_title, fontsize=20, pad=20)
        
        # Save the plot
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def load_model(self, path: str, iteration: int = None) -> None:
        """Load a model from a given path.

        :param path: The path to the model file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        
        pre_trained_path = str(path)
        assert os.path.exists(pre_trained_path), f"Pre-trained weights file not found: {pre_trained_path}"
        if iteration is not None:
            pre_trained_path = os.path.join(pre_trained_path, str(iteration))

        ckpt_files = glob.glob(os.path.join(pre_trained_path, "**", "*.ckpt"), recursive=True)

        ckpt_files = [f for f in ckpt_files if 'last' not in f]
        assert len(ckpt_files) > 0, f"No ckpt files found in {pre_trained_path}"
        ckpt_path = ckpt_files[0]
        print(f"Loading pre-trained weights from {ckpt_path}")
        #ckpt_path = pre_trained_path

    
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback to weights_only=False for older checkpoints with OmegaConf objects
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # Extract state_dict from checkpoint
        state_dict = checkpoint.get('state_dict', checkpoint)
        print(f"Checkpoint keys: {list(state_dict.keys())[:5]}")
        # Extract network weights from the state dict
        # Handle both old format (with 'network.' prefix) and new format
        lc_state_dict = {}
        
        # First, check if we have network keys with prefix
        if self.hparams.dino:
            network_keys = [k for k in state_dict.keys() if k.startswith('student.backbone.')]
        else:
            network_keys = [k for k in state_dict.keys() if k.startswith('network.')]

        if network_keys:
            # Extract network weights by removing 'network.' prefix
            for key, value in state_dict.items():
                if self.hparams.dino:
                    if key.startswith('student.backbone.'):
                        clean_key = key[17:]  # Remove 'student.backbone.' prefix
                        lc_state_dict[clean_key] = value
                else:
                    if key.startswith('network.'):
                        clean_key = key[8:]  # Remove 'network.' prefix
                        lc_state_dict[clean_key] = value
        else:
            # No network prefix, use all keys that don't belong to other components
            for key, value in state_dict.items():
                if not key.startswith('proyector.') and not key.startswith('classifier'):
                    lc_state_dict[key] = value
        
        # Get current model state for comparison
        model_state_dict = self.encoder.state_dict()
        model_keys = set(model_state_dict.keys())
        checkpoint_keys = set(lc_state_dict.keys())
        
        print(f"Model has {len(model_keys)} parameters")
        print(f"Checkpoint has {len(checkpoint_keys)} parameters for network")
        print(f"Sample model keys: {list(model_keys)[:5]}")
        print(f"Sample checkpoint keys: {list(checkpoint_keys)[:5]}")
        
        # Check for architecture compatibility
        missing_in_checkpoint = model_keys - checkpoint_keys
        unexpected_in_checkpoint = checkpoint_keys - model_keys
        
        if len(missing_in_checkpoint) > len(model_keys) * 0.7:
            print(f"WARNING: Too many missing keys ({len(missing_in_checkpoint)}/{len(model_keys)})")
            print("This suggests a major architecture mismatch.")
            print("Skipping checkpoint loading - model will use random initialization.")
            return
        
        try:
            # Try to load the state dict
            missing_keys, unexpected_keys = self.encoder.load_state_dict(lc_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in checkpoint: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    print(f"Unexpected keys: {unexpected_keys}")
                
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print(f"✓ Successfully loaded encoder model from {path}")
            else:
                print(missing_keys)
                print(unexpected_keys)
                print(f"⚠ Partially loaded encoder model from {path}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Checkpoint appears to be from a different model architecture.")
            print("Skipping checkpoint loading - model will use random initialization.")

if __name__ == "__main__":
    _ = EFATATModule(None, None)