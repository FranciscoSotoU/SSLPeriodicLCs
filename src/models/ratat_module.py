from typing import Any, Dict, Tuple
import warnings

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
import torchmetrics
from transformers import get_cosine_schedule_with_warmup
import os
import glob
import seaborn as sns
import numpy as np
from src.models.components.utils.focal_loss import FocalLoss
from src.models.components.utils.middle import Concatenation, ConcatenationRelu, ConcatenationDropOut, ConcatenationKSparse,ConcatenationKSparseTwoTails

# Suppress specific torchmetrics warnings about NaN values in confusion matrix
warnings.filterwarnings("ignore", ".*NaN values found in.*confusion matrix.*", UserWarning)

class ATATLitModule(LightningModule):
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
        use_tabular: bool = False,
        use_lightcurve: bool = True,
        lc_net: torch.nn.Module = None,
        feat_net: torch.nn.Module= None,
        mix_regressor: torch.nn.Module = None,
        feat_regressor: torch.nn.Module = None,
        lc_regressor: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        compile: bool = False,
        freeze_lc_encoder: bool = False,
        freeze_feat_encoder: bool = False,
        pre_trained_lc_path: str = None,
        pre_trained_feat_path: str = None,
        pre_trained_iteration: int = None,
        warmup_epochs: int = 1,
        concatenation_partial: bool = True,
        concatenation_function: str = 'concat',
        UMAP: bool = False,
        dino: bool = False
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['lc_net', 'feat_net', 'mix_regressor', 'feat_regressor', 'lc_regressor'])

        self.lc_net = lc_net if use_lightcurve else None
        self.feat_net = feat_net if use_tabular else None
        self.mix_regressor = mix_regressor if use_lightcurve and use_tabular else None
        
        self.feat_regressor = feat_regressor if use_tabular else None
        self.lc_regressor = lc_regressor  if use_lightcurve else None

        if pre_trained_lc_path is not None and self.lc_net is not None:
            self.load_model_lc(pre_trained_lc_path, iteration=pre_trained_iteration)
        if pre_trained_feat_path is not None and self.feat_net is not None:
            self.load_model_feat(pre_trained_feat_path)
        if self.lc_net is not None and freeze_lc_encoder:
            self.lc_net.requires_grad_(False)
        if self.feat_net is not None and freeze_feat_encoder:
            self.feat_net.requires_grad_(False)
        # loss function
        self.criterion = torch.nn.MSELoss()
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_loss_lc = MeanMetric()
        self.train_loss_feat = MeanMetric()
        self.train_loss_mix = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_lc = MeanMetric()
        self.val_loss_feat = MeanMetric()
        self.val_loss_mix = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_loss_lc = MeanMetric()
        self.test_loss_feat = MeanMetric()
        self.test_loss_mix = MeanMetric()
        #get best loss and f1 score
        self.val_loss_min = MinMetric()
        concatenations_dict = {
            'concat': Concatenation,
            'relu': ConcatenationRelu,
            'dropout': ConcatenationDropOut,
            'k_sparse': ConcatenationKSparse,
            'k_sparse_tt': ConcatenationKSparseTwoTails,
        }
        self.concat_function = concatenations_dict.get(concatenation_function)()
        self.UMAP = UMAP 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        lc_branch_out = None
        feat_branch_out = None
        mix_out = None
        if self.hparams.freeze_lc_encoder and self.lc_net is not None:
            self.lc_net.eval()
        if self.lc_net is not None:
            lc_out = self.lc_net(**x)
            lc_branch_out = self.lc_regressor(lc_out)
        if self.feat_net is not None:
            feat_out = self.feat_net(**x)
            feat_branch_out = self.feat_regressor(feat_out)
        if self.lc_net is not None and self.feat_net is not None:
            mix_token = self.concat_function([lc_out, feat_out])
            mix_out = self.mix_regressor(mix_token)
        return lc_branch_out,feat_branch_out, mix_out
    def forward_lattent(self, x: torch.Tensor):
        """Perform a forward pass through the model `self.net` and return the latent representations.

        :param x: A tensor of images.
        :return: A tuple of latent representations from lc_net and feat_net.
        """
        lc_branch_out = None
        feat_branch_out = None
        if self.lc_net is not None:
            lc_out = self.lc_net(**x)
            lc_branch_out = lc_out
        if self.feat_net is not None:
            feat_out = self.feat_net(**x)
            feat_branch_out = feat_out
        return lc_branch_out,feat_branch_out
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_lc.reset()
        self.val_loss_feat.reset()
        self.val_loss_min.reset()


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
        lcs_preds,feat_preds,mix_preds = self.forward(batch)
        loss = 0.0
        
        # Collect predictions for main metrics
        main_preds = None
        
        if self.lc_net is not None:
            lc_loss = self.criterion(lcs_preds, y)
            loss += lc_loss
            lcs_preds_argmax = torch.argmax(lcs_preds, dim=1)
            self.train_loss_lc(lc_loss)
            self.train_metrics_lc(lcs_preds_argmax, y)
            self.log("train/loss_lc", self.train_loss_lc, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.train_metrics_lc, on_step=False, on_epoch=True, prog_bar=True)
            
            # Use lc predictions for main metrics if mix is not available
            if main_preds is None:
                main_preds = lcs_preds_argmax
                
        if self.feat_net is not None:
            feat_loss = self.criterion(feat_preds, y)
            loss += feat_loss
            feat_preds_argmax = torch.argmax(feat_preds, dim=1)
            self.train_loss_feat(feat_loss)
            self.train_metrics_feat(feat_preds_argmax, y)
            self.log("train/loss_feat", self.train_loss_feat, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.train_metrics_feat, on_step=False, on_epoch=True, prog_bar=True)
            
            # Use feat predictions for main metrics if mix is not available and lc is not available
            if main_preds is None:
                main_preds = feat_preds_argmax
                
        if self.mix_classifier is not None:
            mix_loss = self.criterion(mix_preds, y)
            loss += mix_loss
            mix_preds_argmax = torch.argmax(mix_preds, dim=1)
            self.train_loss_mix(mix_loss)
            self.train_metrics_mix(mix_preds_argmax, y)
            self.log("train/loss_mix", self.train_loss_mix, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.train_metrics_mix, on_step=False, on_epoch=True, prog_bar=True)
            
            # Use mix predictions for main metrics (highest priority)
            main_preds = mix_preds_argmax
            
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update main metrics with the appropriate predictions
        if main_preds is not None:
            self.train_metrics(main_preds, y)
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
        y = batch['label'].long()
        lcs_preds,feat_preds,mix_preds = self.forward(batch)
        loss = 0.0
        
        # Collect predictions for main metrics
        main_preds = None
        
        if self.lc_net is not None:
            lc_loss = self.criterion(lcs_preds, y)
            loss += lc_loss
            lcs_preds_argmax = torch.argmax(lcs_preds, dim=1)
            self.val_loss_lc(lc_loss)
            self.val_metrics_lc(lcs_preds_argmax, y)
            self.val_cm_lc(lcs_preds_argmax, y)
            self.log("val/loss_lc", self.val_loss_lc, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.val_metrics_lc, on_step=False, on_epoch=True, prog_bar=True)
            
            # Use lc predictions for main metrics if mix is not available
            if main_preds is None:
                main_preds = lcs_preds_argmax
                
        if self.feat_net is not None:
            feat_loss = self.criterion(feat_preds, y)
            loss += feat_loss
            feat_preds_argmax = torch.argmax(feat_preds, dim=1)
            self.val_loss_feat(feat_loss)
            self.val_metrics_feat(feat_preds_argmax, y)
            self.val_cm_feat(feat_preds_argmax, y)
            self.log("val/loss_feat", self.val_loss_feat, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.val_metrics_feat, on_step=False, on_epoch=True, prog_bar=True)
            
            # Use feat predictions for main metrics if mix is not available and lc is not available
            if main_preds is None:
                main_preds = feat_preds_argmax
                
        if self.mix_classifier is not None:
            mix_loss = self.criterion(mix_preds, y)
            loss += mix_loss
            mix_preds_argmax = torch.argmax(mix_preds, dim=1)
            self.val_loss_mix(mix_loss)
            self.val_metrics_mix(mix_preds_argmax, y)
            self.val_cm(mix_preds_argmax, y)
            self.log("val/loss_mix", self.val_loss_mix, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.val_metrics_mix, on_step=False, on_epoch=True, prog_bar=True)
            
            # Use mix predictions for main metrics (highest priority)
            main_preds = mix_preds_argmax

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update main metrics with the appropriate predictions
        if main_preds is not None:
            self.val_metrics(main_preds, y)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # Only plot confusion matrices for lc and feat if we have a new best mix loss
        if self.mix_classifier is not None:
            current_val_loss = self.val_loss.compute()
            # Update best loss and check if this is a new best
            self.val_loss_min.update(current_val_loss)
            new_best = self.val_loss_min.compute()
            
            # Plot epoch matrices only if plot_all_epochs_cms is True
            if self.hparams.plot_all_epochs_cms:
                self.plot_confusion_matrix(stage='val', branch='mix', epoch=self.current_epoch, best=False)
            
            # If we achieved a new best loss, save all matrices as best
            if current_val_loss <= new_best:
                # Save best matrices in main cms folder
                self.plot_confusion_matrix(stage='val', branch='mix', epoch=self.current_epoch, best=True)
                if self.lc_net is not None:
                    self.plot_confusion_matrix(stage='val', branch='lc', epoch=self.current_epoch, best=True)
                if self.feat_net is not None:
                    self.plot_confusion_matrix(stage='val', branch='feat', epoch=self.current_epoch, best=True)
            else:
                # Save regular epoch matrices in epochs subfolder only if plot_all_epochs_cms is True
                if self.hparams.plot_all_epochs_cms:
                    if self.lc_net is not None:
                        self.plot_confusion_matrix(stage='val', branch='lc', epoch=self.current_epoch, best=False)
                    if self.feat_net is not None:
                        self.plot_confusion_matrix(stage='val', branch='feat', epoch=self.current_epoch, best=False)
        else:
            # If no mix classifier, check individual branches for best
            current_val_loss = self.val_loss.compute()
            self.val_loss_min.update(current_val_loss)
            new_best = self.val_loss_min.compute()
            
            is_best = current_val_loss <= new_best
            
            # Save confusion matrices
            if self.lc_net is not None:
                if is_best:
                    # Save best matrix in main cms folder
                    self.plot_confusion_matrix(stage='val', branch='lc', epoch=self.current_epoch, best=True)
                # Save epoch matrix in epochs subfolder only if plot_all_epochs_cms is True
                if self.hparams.plot_all_epochs_cms:
                    self.plot_confusion_matrix(stage='val', branch='lc', epoch=self.current_epoch, best=False)
            
            if self.feat_net is not None:
                if is_best:
                    # Save best matrix in main cms folder
                    self.plot_confusion_matrix(stage='val', branch='feat', epoch=self.current_epoch, best=True)
                # Save epoch matrix in epochs subfolder only if plot_all_epochs_cms is True
                if self.hparams.plot_all_epochs_cms:
                    self.plot_confusion_matrix(stage='val', branch='feat', epoch=self.current_epoch, best=False)
                    
        # Reset confusion matrices for the next epoch
        self.val_cm.reset()
        self.val_cm_lc.reset()
        self.val_cm_feat.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        y = batch['label'].long()
        lcs_preds,feat_preds,mix_preds = self.forward(batch)
        loss = 0.0

        main_preds = None

        if self.lc_net is not None:
            lc_loss = self.criterion(lcs_preds, y)
            loss += lc_loss
            lcs_preds = torch.argmax(lcs_preds, dim=1)
            self.test_loss_lc(lc_loss)
            self.test_metrics_lc(lcs_preds, y)
            self.test_cm_lc(lcs_preds, y)
            self.log("test/loss_lc", self.test_loss_lc, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.test_metrics_lc, on_step=False, on_epoch=True, prog_bar=True)

            # Use lc predictions for main metrics if mix is not available
            if main_preds is None:
                main_preds = lcs_preds

        if self.feat_net is not None:
            feat_loss = self.criterion(feat_preds, y)
            loss += feat_loss
            feat_preds = torch.argmax(feat_preds, dim=1)
            self.test_loss_feat(feat_loss)
            self.test_metrics_feat(feat_preds, y)
            self.test_cm_feat(feat_preds, y)
            self.log("test/loss_feat", self.test_loss_feat, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.test_metrics_feat, on_step=False, on_epoch=True, prog_bar=True)

            if main_preds is None:
                main_preds = feat_preds

        if self.mix_classifier is not None:
            mix_loss = self.criterion(mix_preds, y)
            loss += mix_loss
            mix_preds = torch.argmax(mix_preds, dim=1)
            self.test_loss_mix(mix_loss)
            self.test_metrics_mix(mix_preds, y)
            self.test_cm(mix_preds, y)
            self.log("test/loss_mix", self.test_loss_mix, on_step=False, on_epoch=True, prog_bar=True)
            self.log_dict(self.test_metrics_mix, on_step=False, on_epoch=True, prog_bar=True)

        # Update main metrics with the appropriate predictions
            main_preds = mix_preds
        self.test_loss(loss)
        # log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Update main metrics with the appropriate predictions
        if main_preds is not None:
            self.test_metrics(main_preds, y)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Plot confusion matrices for all available branches
        if self.mix_classifier is not None:
            self.plot_confusion_matrix(stage='test', branch='mix')
        
        if self.lc_net is not None:
            self.plot_confusion_matrix(stage='test', branch='lc')
        
        if self.feat_net is not None:
            self.plot_confusion_matrix(stage='test', branch='feat')

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single predict step on a batch of data from the predict set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of thse current batch.
        """

        

        oid = batch['oid']
        targets = batch['label'].long()
        
        if self.UMAP:
            # If UMAP is enabled, we only return the latent representations
            lcs,feat = self.forward_lattent(batch)
            # Convert to numpy arrays for UMAP processing
            lcs = lcs if lcs is not None else None
            feat = feat if feat is not None else None
            return {
                "latent_lc": lcs,
                "latent_feat": feat,
                "oid": oid,
                "targets": targets,
                
            }
        else:
            lcs,feat,mix = self.forward(batch)
  
        return {
            "logits_lc": lcs,
            "logits_feat": feat,
            "logits_mix": mix,
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
            if self.lc_net is not None:
                self.lc_net = torch.compile(self.lc_net)
            if self.lc_classifier is not None:
                self.lc_classifier = torch.compile(self.lc_classifier)
            if self.feat_net is not None:
                self.feat_net = torch.compile(self.feat_net)
            if self.feat_classifier is not None:
                self.feat_classifier = torch.compile(self.feat_classifier)

            
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
    def plot_confusion_matrix(self, stage='val', branch='mix', epoch=None, best=False) -> str:
        """
        Create and save a confusion matrix plot for the specified stage and branch.
        
        Args:
            stage: Stage name ('val', 'test')
            branch: Branch type ('mix', 'lc', 'feat')
            epoch: Current epoch number (None for test)
        
        Returns:
            str: Path where the confusion matrix was saved
        """
        import matplotlib.pyplot as plt
        
        # Get the appropriate confusion matrix and F1 score
        if stage == 'val':
            if branch == 'mix':
                cm = self.val_cm.compute()
                current_f1 = self.val_metrics['val/f1'].compute()
            elif branch == 'lc':
                cm = self.val_cm_lc.compute()
                current_f1 = self.val_metrics_lc['val/f1_lc'].compute()
            elif branch == 'feat':
                cm = self.val_cm_feat.compute()
                current_f1 = self.val_metrics_feat['val/f1_feat'].compute()
        else:  # test
            if branch == 'mix':
                cm = self.test_cm.compute()
                current_f1 = self.test_metrics['test/f1'].compute()
            elif branch == 'lc':
                cm = self.test_cm_lc.compute()
                current_f1 = self.test_metrics_lc['test/f1_lc'].compute()
            elif branch == 'feat':
                cm = self.test_cm_feat.compute()
                current_f1 = self.test_metrics_feat['test/f1_feat'].compute()
        
        # Manually normalize the confusion matrix to avoid NaN issues
        cm_normalized  =cm.float()
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
        
        # Adjust styling based on stage
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
        
        # Create title and filename based on stage and branch
        branch_names = {'mix': 'Mixed', 'lc': 'Light Curve', 'feat': 'Features'}
        branch_display = branch_names.get(branch, branch.upper())
        
        stage_str = stage.capitalize()
        if stage == 'test':
            cm_title = f'Confusion Matrix - {stage_str} ({branch_display})\nF1 Score: {current_f1:.3f}'
            filename = f'confusion_matrix_{stage}_{branch}.png'
            save_dir = os.path.join(self.trainer.default_root_dir, 'cms')
        else:
            if best:
                cm_title = f'Confusion Matrix - Best {stage_str} ({branch_display})\nF1 Score: {current_f1:.3f}'
                filename = f'confusion_matrix_best_{stage}_{branch}.png'
                save_dir = os.path.join(self.trainer.default_root_dir, 'cms')
            else:
                cm_title = f'Confusion Matrix - Epoch {epoch} {stage_str} ({branch_display})\nF1 Score: {current_f1:.3f}'
                filename = f'confusion_matrix_epoch_{epoch:03d}_{stage}_{branch}.png'
                save_dir = os.path.join(self.trainer.default_root_dir, 'cms', 'epochs')
        plt.title(cm_title, fontsize=20, pad=20)
        
        # Save the plot
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    def load_model_lc(self, path: str, iteration: int = None) -> None:
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
        model_state_dict = self.lc_net.state_dict()
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
            missing_keys, unexpected_keys = self.lc_net.load_state_dict(lc_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in checkpoint: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    print(f"Unexpected keys: {unexpected_keys}")
                
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                
                print(f"✓ Successfully loaded lightcurve model from {path}")
            else:
                print(missing_keys)
                print(unexpected_keys)
                print(f"⚠ Partially loaded lightcurve model from {path}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Checkpoint appears to be from a different model architecture.")
            print("Skipping checkpoint loading - model will use random initialization.")

    def load_model_feat(self, path: str) -> None:
        """Load a model from a given path.

        :param path: The path to the model file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        try:
            # First try with weights_only=True (safer)
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback to weights_only=False for older checkpoints with OmegaConf objects
            state_dict = torch.load(path, map_location=self.device, weights_only=False)
        
        network = state_dict.get('network', state_dict)
        if 'feat_net' in network:
            self.feat_net.load_state_dict(network['feat_net'])
        else:
            raise KeyError("No 'feat_net' found in the state dictionary.")

if __name__ == "__main__":
    _ = ATATLitModule(None, None, None,None)