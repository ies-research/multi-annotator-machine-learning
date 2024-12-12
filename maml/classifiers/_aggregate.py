import torch

from torch import nn
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import functional as F
from typing import Optional, Dict

from ..classifiers import MaMLClassifier
from ..utils import mixup


class AggregateClassifier(MaMLClassifier):
    """AggregateClassifier

    This classifier is a general implementation for two-stage training procedures, where annotations are aggregated
    as ground truth (GT) estimates in the first stage and used for training the GT model in the second stage.
    Approaches using a two-stage training procedure do not support an annotator performance (AP) model.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    gt_embed_x : nn.Module
        Pytorch module the GT model's backbone embedding the input samples.
    gt_output : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
    alpha : float, default=0.0
        Determines the parameters of the beta distribution used for sampling the mixup coefficients.
    optimizer : torch.optim.Optimizer.__class__, optional (default=RAdam.__class__)
        Optimizer class responsible for optimizing the GT and AP parameters. If `None`, the `RAdam` optimizer is used
        by default.
    optimizer_gt_dict : dict, optional (default=None)
        Parameters passed to `optimizer` for the GT model.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler.__class__, optional (default=None)
        Learning rate scheduler responsible for optimizing the GT and AP parameters. If `None`, no learning rate
        scheduler is used by default.
    lr_scheduler_dict : dict, optional (default=None)
        Parameters passed to `lr_scheduler`.
    """

    def __init__(
        self,
        n_classes: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        alpha: float = 0.0,
        optimizer: Optional[Optimizer.__class__] = RAdam,
        optimizer_gt_dict: Optional[Dict] = None,
        lr_scheduler: Optional[LRScheduler.__class__] = None,
        lr_scheduler_dict: Optional[Dict] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_gt_dict=optimizer_gt_dict,
            optimizer_ap_dict=None,
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
        """Forward propagation of samples through the GT model.

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
        """
        Computes the GT models' loss.

        Parameters
        ----------
        batch : dict
            Data batch fitting the dictionary structure of `maml.data.MultiAnnotatorDataset`.
        batch_idx : int
            Index of the batch in the dataset.
        dataloader_idx : int, default=0
            Index of the used dataloader.

        Returns
        -------
        loss : torch.Float
            Computed cross-entropy loss.
        """
        x, z_agg = batch["x"], batch["z_agg"]

        # One-hot encode aggregated annotations.
        if z_agg.ndim != 2:
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
        """
        Computes the GT models' predictions.

        Parameters
        ----------
        batch : dict
            Data batch fitting the dictionary structure of `maml.data.MultiAnnotatorDataset`.
        batch_idx : int
            Index of the batch in the dataset.
        dataloader_idx : int, default=0
            Index of the used dataloader.

        Returns
        -------
        predictions : dict
            A dictionary of predictions fitting the expected structure of `maml.classifiers.MaMLClassifier`.
        """
        return {"p_class": self.forward(x=batch["x"]).softmax(dim=-1)}

    @staticmethod
    def loss(z_agg: torch.tensor, logits_class: torch.tensor):
        """
        Computes the cross entropy between the aggregated annotations and the estimated class-membership probabilities.

        Parameters
        ----------
        z_agg : torch.tensor of shape (batch_size,)
            Aggregated annotations for the current batch.
        logits_class : torch.tensor of shape (batch_size, n_classes)
            Logits outputted by the GT model's last layer.

        Returns
        -------
        loss_gt : torch.Float
            Computed cross-entropy loss.
        """
        is_labeled = z_agg.sum(dim=-1) > 0
        n_labels = is_labeled.sum().float().item()

        # Compute prediction loss for GT model.
        p_class_log = logits_class.log_softmax(dim=-1)
        loss_gt = -(z_agg * p_class_log).sum()
        return loss_gt / n_labels

    @torch.no_grad()
    def get_gt_parameters(self):
        """Returns the list of parameters of the GT model."""
        gt_parameters = list(self.gt_embed_x.parameters())
        gt_parameters += list(self.gt_output.parameters())
        return gt_parameters

    @torch.no_grad()
    def get_ap_parameters(self):
        """
        Returns the list of parameters of the AP model. Since two-stage training procedures do not support an AP
        model, the list of parameters is empty.
        """
        return []
