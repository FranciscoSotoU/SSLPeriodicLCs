from typing import Any, Dict, Tuple, Optional
import math
import copy
import numpy as np
import torch
import torch.nn as nn

from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

from src.models.components.multicrop_wrapper import MultiCropWrapper


class DINOLitModule(LightningModule):
    """DINO (Self-Distillation with No Labels) Lightning Module.

    DINO uses a teacher-student framework where:
    - Student network is trained with gradient descent
    - Teacher network is an exponential moving average of the student
    - Teacher predictions are centered to avoid collapse
    - Cross-entropy loss between teacher and student predictions

    Paper: "Emerging Properties in Self-Supervised Vision Transformers"
    """

    def __init__(
        self,
        network: torch.nn.Module,
        head_teacher: torch.nn.Module,
        head_student: torch.nn.Module,
        student_temp: float = 0.1,
        teacher_temp: float = 0.07,
        warmup_teacher_temp: float = 0.04,
        teacher_temp_warmup_epochs: int = 30,
        momentum_teacher: float = 0.996,
        compile: bool = False,
        optimizer: torch.optim.Optimizer = None,
        warmup_epochs: int = 10,
        center_momentum: float = 0.9,
        max_length: int = 512,
        num_crops: int = 10,  # Add this parameter
        freeze_last_layer: int = 1,
        **kwargs
    ) -> None:
        """Initialize DINO module."""
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(
            logger=False, ignore=["network", "head_teacher", "head_student"]
        )

        # Student networks
        self.student = MultiCropWrapper(network, head_student)

        # Teacher networks (EMA of student)
        self.teacher = MultiCropWrapper(copy.deepcopy(network), head_teacher)

        # Disable gradients for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Initialize teacher with student weights (CRITICAL FIX)
        self.update_teacher(momentum=0.0)  # Full copy at initialization

        projection_size = head_teacher.projection_size

        # Center for teacher outputs to avoid collapse
        self.register_buffer("center", torch.zeros(1, projection_size))
        self.center_momentum = center_momentum
        self.momentum_teacher = momentum_teacher

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor, use_teacher: bool = False) -> torch.Tensor:
        """Forward pass through student or teacher network."""
        if use_teacher:
            return self.teacher(x[:2])  # Only global crops for teacher
        else:
            return self.student(x)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    def dino_loss(
        self, student_output: torch.Tensor, teacher_output: torch.Tensor
    ) -> torch.Tensor:
        """Compute DINO loss between student and teacher predictions."""
        # Apply temperature scaling to student
        student_out = student_output / self.hparams.student_temp
        student_out = student_out.chunk(self.hparams.num_crops)

        # Apply temperature scaling to teacher with centering
        teacher_temp = self.teacher_temp_schedule[self.current_epoch]
        teacher_out = F.softmax(
            (teacher_output - self.center) / teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach().chunk(2)  # Only 2 global crops

        total_loss = 0
        n_loss_terms = 0

        # Cross-entropy loss between all student crops and teacher crops
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # Skip same view
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center after loss computation
        self.update_center(teacher_output)
        return total_loss

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single model step on a batch of data."""
        student_output = self.forward(batch, use_teacher=False)
        teacher_output = self.forward(batch, use_teacher=True)

        loss = self.dino_loss(student_output, teacher_output)
        return loss, student_output, teacher_output

    @torch.no_grad()
    def update_teacher(self, momentum):
        """Update teacher network with EMA of student."""
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step."""

        # Compute loss
        loss, student_output, teacher_output = self.model_step(batch)
        batch_size = batch[0]["data"].shape[0]

        # Log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/center_norm",
            torch.norm(self.center),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log("train/student_std", torch.std(student_output), on_step=True)
        self.log("train/teacher_std", torch.std(teacher_output), on_step=True)
        self.log("train/teacher_mean", torch.mean(teacher_output), on_step=True)
        self.log("train/student_mean", torch.mean(student_output), on_step=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True)
        self.log(
            "train/teacher_temp",
            self.teacher_temp_schedule[self.current_epoch],
            on_step=True,
        )
        self.log(
            "train/momentum_teacher",
            self.teacher_momentum_scheduler[
                self.current_epoch * self.num_training_batches + batch_idx
            ],
            on_step=True,
        )
        if self.hparams.freeze_last_layer > 0:
            self.cancel_gradients_last_layer(
                self.current_epoch, self.student, self.hparams.freeze_last_layer
            )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_teacher(
            self.teacher_momentum_scheduler[
                self.current_epoch * self.num_training_batches + batch_idx
            ]
        )

    def cancel_gradients_last_layer(self, epoch, model, freeze_last_layer):
        if epoch >= freeze_last_layer:
            return
        for n, p in model.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step."""
        loss, student_output, teacher_output = self.model_step(batch)

        batch_size = batch[0]["data"].shape[0]

        # Log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def on_train_start(self) -> None:
        """Lightning hook called when training begins."""
        self.val_loss.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook called at the beginning of fit, validate, test, or predict."""
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()
            self.num_training_batches = len(train_loader)
            self.teacher_momentum_scheduler = cosine_scheduler(
                base_value=self.momentum_teacher,
                final_value=1.0,
                epochs=self.trainer.max_epochs,
                niter_per_ep=len(train_loader),
            )
            self.teacher_temp_schedule = np.concatenate(
                (
                    np.linspace(
                        self.hparams.warmup_teacher_temp,
                        self.hparams.teacher_temp,
                        self.hparams.teacher_temp_warmup_epochs,
                    ),
                    np.ones(self.trainer.max_epochs - self.hparams.teacher_temp_warmup_epochs)
                    * self.hparams.teacher_temp,
                )
            )
            if self.hparams.compile and (stage == "fit" or stage == "predict"):
                self.student = torch.compile(self.student)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Only optimize student parameters
        student_params = self.student.parameters()
        optimizer = self.hparams.optimizer(student_params)

        num_training_batches = self.num_training_batches
        warmup_steps = self.hparams.warmup_epochs * num_training_batches
        total_steps = self.trainer.max_epochs * num_training_batches
        min_lr_ratio = 0.1  # Minimum LR as 1% of initial LR

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Warmup phase
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing phase
                progress = float(
                    current_step - warmup_steps
                ) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_schedule_with_min_lr",
            },
        }


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
