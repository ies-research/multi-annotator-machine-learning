import torch

from abc import ABC, abstractmethod
from lightning.pytorch import LightningModule
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict


class MaMLClassifier(LightningModule, ABC):
    """MaMLClassifier

    This is the base class for implementing "multi-annotator machine learning" (MAML) models. Thereby, the basic
    assumption is that such a model consists of a ground truth (GT) model estimating the ground truth of each sample
    and an annotator performance (AP) model estimating the performance of annotators.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer.__class__, optional (default=RAdam.__class__)
        Optimizer class responsible for optimizing the GT and AP parameters. If `None`, the `RAdam` optimizer is used
        by default.
    optimizer_gt_dict : dict, optional (default=None)
        Parameters passed to `optimizer` for the GT model.
    optimizer_ap_dict : dict, optional (default=None)
        Parameters passed to `optimizer` for the AP model.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler.__class__, optional (default=None)
        Learning rate scheduler responsible for optimizing the GT and AP parameters. If `None`, no learning rate
        scheduler is used by default.
    lr_scheduler_dict : dict, optional (default=None)
        Parameters passed to `lr_scheduler`.
    """

    def __init__(
        self,
        optimizer: Optimizer.__class__ = RAdam,
        optimizer_gt_dict: Optional[dict] = None,
        optimizer_ap_dict: Optional[dict] = None,
        lr_scheduler: Optional[LRScheduler.__class__] = None,
        lr_scheduler_dict: Optional[dict] = None,
    ):
        super().__init__()

        # Define optimizer and its optional parameters.
        self.optimizer = optimizer
        self.optimizer_gt_dict = optimizer_gt_dict
        self.optimizer_ap_dict = optimizer_ap_dict

        # Define learning rate scheduler and its optional parameters.
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_dict = lr_scheduler_dict

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: int = 0):
        """
        Evaluates and logs the performance of the GT model on the given validation data as `"gt_val_acc"`.

        Parameters
        ----------
        batch : dict
            Data batch fitting the dictionary structure of `maml.data.MultiAnnotatorDataset`.
        batch_idx : int
            Index of the batch in the dataset.
        dataloader_idx : int, default=0
            Index of the used dataloader.
        """
        batch.pop("a", None)
        output = self.predict_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        p_class = output.get("p_class")
        y_pred = p_class.argmax(dim=-1)
        acc = (y_pred == batch["y"]).float().mean()
        self.log("gt_val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y_pred))

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        """
        Evaluates and logs the performance of the GT model on the given test data as `"gt_val_acc"` and (optionally)
        of the AP model as "ap_test_acc".

        Parameters
        ----------
        batch : dict
            Data batch fitting the dictionary structure of `maml.data.MultiAnnotatorDataset`.
        batch_idx : int
            Index of the batch in the dataset.
        dataloader_idx : int, default=0
            Index of the used dataloader.
        """
        output = self.predict_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        p_class = output.get("p_class")
        p_perf = output.get("p_perf", None)

        # Evaluate GT model accuracy.
        y_gt_pred = p_class.argmax(dim=-1)
        y = batch["y"]
        gt_test_acc = (y_gt_pred == y).float().mean()
        self.log_dict({"gt_test_acc": gt_test_acc}, batch_size=len(y), on_step=False, on_epoch=True)

        # Evaluate AP model accuracy.
        z = batch.get("z", None)
        if p_perf is not None and z is not None:
            is_labeled = z != -1
            is_true = (y[:, None] == z).long()
            is_true = is_true.ravel()[is_labeled.ravel()]
            p_perf = p_perf.ravel()[is_labeled.ravel()]
            y_ap_pred = (p_perf > 0.5).long()
            ap_test_acc = ((y_ap_pred == is_true).float()).mean()
            self.log_dict({"ap_test_acc": ap_test_acc}, batch_size=len(is_true), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers of the ground truth (GT) and annotator performance (AP)
        models.

        Returns
        -------
        optimizers : list
            The list of configured optimizers.
        lr_schedulers : list
            The list of configured learning rate schedulers.
        """
        # Setup optimizer.
        optimizer_gt_dict = {} if self.optimizer_gt_dict is None else self.optimizer_gt_dict
        optimizer_ap_dict = optimizer_gt_dict if self.optimizer_ap_dict is None else self.optimizer_ap_dict
        optimizer = self.optimizer(
            [
                {"params": self.get_gt_parameters(), **optimizer_gt_dict},
                {"params": self.get_ap_parameters(), **optimizer_ap_dict},
            ]
        )

        # Return optimizer, if no learning rate scheduler has been defined.
        if self.lr_scheduler is None:
            return [optimizer]

        lr_scheduler_dict = {} if self.lr_scheduler_dict is None else self.lr_scheduler_dict
        lr_scheduler = self.lr_scheduler(optimizer, **lr_scheduler_dict)
        return [optimizer], [lr_scheduler]

    @abstractmethod
    def get_gt_parameters(self, **kwargs):
        """Returns the list of parameters of the GT model."""
        pass

    @abstractmethod
    def get_ap_parameters(self, **kwargs):
        """Returns the list of parameters of the AP model."""
        pass
