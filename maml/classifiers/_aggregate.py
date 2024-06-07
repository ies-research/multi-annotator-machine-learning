import torch

from torch import nn
from torch import optim
from torch.nn import functional as F
from typing import Optional, Dict

from ..classifiers import MaMLClassifier
from ..utils import mixup


class AggregateClassifier(MaMLClassifier):
    def __init__(
        self,
        n_classes: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        alpha: float = 0.0,
        optimizer: Optional[optim.Optimizer.__class__] = None,
        optimizer_gt_dict: Optional[Dict] = None,
        optimizer_ap_dict: Optional[Dict] = None,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler.__class__] = None,
        lr_scheduler_dict: Optional[Dict] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_gt_dict=optimizer_gt_dict,
            optimizer_ap_dict=optimizer_ap_dict,
            lr_scheduler=lr_scheduler,
            lr_scheduler_dict=lr_scheduler_dict,
        )
        self.n_classes = n_classes
        self.alpha = alpha
        self.gt_embed_x = gt_embed_x
        self.gt_output = gt_output

        # Save hyper parameters.
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.tensor):
        """Forward propagation of samples' and annotators' (optional) features through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.tensor of shape (batch_size, *)
            Sample features.

        Returns
        -------
        logits_class : torch.tensor of shape (batch_size, n_classes)
            Class-membership logits.
        """
        # Compute feature embedding for classifier.
        x_learned = self.gt_embed_x(x)

        # Compute class-membership probabilities.
        logits_class = self.gt_output(x_learned)

        return logits_class

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        x, z_agg = batch["x"],  batch["z_agg"]

        # One-hot encode aggregated annotations.
        z_agg = F.one_hot(z_agg + 1, num_classes=self.n_classes + 1)[:, 1:].float()

        # Perform mixup, if required.
        if self.alpha > 0:
            x, z_agg, lmbda, index = mixup(x, z_agg, alpha=self.alpha)

        # Forward propagation.
        logits_class = self.forward(x=x)

        # Compute loss.
        loss = AggregateClassifier.loss(
            z_agg=z_agg,
            logits_class=logits_class,
        )
        return loss

    @torch.inference_mode()
    def predict_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        p_class = self.forward(x=batch["x"]).softmax(dim=-1)
        return {"p_class": p_class}

    @staticmethod
    def loss(z_agg: torch.tensor, logits_class: torch.tensor):
        is_labeled = z_agg.sum(dim=-1) > 0
        n_labels = is_labeled.sum().float().item()

        # Compute prediction loss for GT model.
        p_class_log = logits_class.log_softmax(dim=-1)
        loss_gt = -(z_agg * p_class_log).sum()
        return loss_gt / n_labels


    @torch.no_grad()
    def get_gt_parameters(self):
        gt_parameters = list(self.gt_embed_x.parameters())
        gt_parameters += list(self.gt_output.parameters())
        return gt_parameters

    @torch.no_grad()
    def get_ap_parameters(self):
        return []
