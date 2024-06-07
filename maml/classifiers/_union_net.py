import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict

from maml.classifiers._base import MaMLClassifier


class UnionNetClassifier(MaMLClassifier):
    """UnionNetClassifier

    UnionNet (Variant B) [1] concatenates the one-hot encoded vectors of labels provided by all annotators. As a
    result, all annotation information are taken as a union into account.


    Parameters
    ----------
    n_classes : int
        Number of classes.
    n_annotators : int
        Number of annotators.
    gt_embed_x : nn.Module
        Pytorch module of the GT model embedding the input samples.
    gt_output : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
    epsilon : non-negative float, optional (default=1e-5)
        Prior error probability to initialize annotators' confusion matrices.
    optimizer : torch.optim.Optimizer, optional (default=None)
        Optimizer responsible for optimizing the GT and AP parameters. If None, the `AdamW` optimizer is used by
        default.
    optimizer_dict : dict, optional (default=None)
        Parameters passed to `optimizer`.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler, optional (default=None)
        Optimizer responsible for optimizing the GT and AP parameters. If None, the `AdamW` optimizer is used by
        default.
    lr_scheduler_dict : dict, optional (default=None)
        Parameters passed to `lr_scheduler`.

    References
    ----------
    [1] Wei, Hongxin, Renchunzi Xie, Lei Feng, Bo Han, and Bo An. "Deep Learning From Multiple Noisy Annotators as A
        Union." IEEE Transactions on Neural Networks and Learning Systems (2022).
    """

    def __init__(
        self,
        n_classes: int,
        n_annotators: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        epsilon: Optional[float] = 1e-5,
        optimizer: Optional[Optimizer.__class__] = AdamW,
        optimizer_gt_dict: Optional[dict] = None,
        optimizer_ap_dict: Optional[dict] = None,
        lr_scheduler: Optional[LRScheduler.__class__] = None,
        lr_scheduler_dict: Optional[dict] = None,
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_gt_dict=optimizer_gt_dict,
            optimizer_ap_dict=optimizer_ap_dict,
            lr_scheduler=lr_scheduler,
            lr_scheduler_dict=lr_scheduler_dict,
        )
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_embed_x = gt_embed_x
        self.gt_output = gt_output
        self.epsilon = epsilon

        # Create transition matrix: cf. Eq. (7) in the article [1].
        init_matrix = (1 - epsilon) * torch.eye(n_classes) + epsilon / (n_classes - 1) * (1 - torch.eye(n_classes))

        # Concatenate transition matrices: cf. Section IV.C in the article [1].
        self.transition_matrix = nn.Parameter(torch.concat([init_matrix for _ in range(n_annotators)]))

        # Store hyperparameters to reload weights.
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.tensor, return_ap_outputs: bool = False):
        """Forward propagation of samples through the GT model.

        Parameters
        ----------
        x : torch.tensor of shape (batch_size, *)
            Sample features.
        return_ap_outputs: bool, optional (default=True)
            Flag whether the annotation probabilities are to be additionally returned.

        Returns
        -------
        logits_class : torch.tensor of shape (batch_size, n_classes)
            Class-membership logits.
        """
        # Compute feature embedding for classifier.
        x_learned = self.gt_embed_x(x)

        # Compute class-membership probabilities.
        logits_class = self.gt_output(x_learned)

        # Compute class-membership probabilities.
        p_class = F.softmax(logits_class, dim=-1)

        if not return_ap_outputs:
            return p_class

        # Compute logits per annotator: cf. input of logarithm function of Eq. (6) in the article [1].
        p_union = p_class @ F.softmax(self.transition_matrix, dim=0).T

        return p_class, p_union

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        _, p_union = self.forward(x=batch["x"], return_ap_outputs=True)
        loss = UnionNetClassifier.loss(
            z=batch["z"],
            p_union=p_union,
            n_classes=self.n_classes,
        )
        return loss

    @torch.inference_mode()
    def predict_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        self.eval()
        a = batch.get("a", None)
        if a is None:
            p_class = self.forward(x=batch["x"], return_ap_outputs=False)
            return {"p_class": p_class}
        else:
            p_class, p_union = self.forward(x=batch["x"], return_ap_outputs=True)
            p_annot = p_union.reshape(-1, self.n_annotators, self.n_classes)
            p_annot /= p_annot.sum(dim=-1, keepdims=True)
            p_perf = torch.stack([torch.einsum("ij,ik->ijk", p_class, p_annot[:, i, :]) for i in a[0]])
            p_perf = p_perf.swapaxes(0, 1).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            return {"p_class": p_class, "p_perf": p_perf}

    @staticmethod
    def loss(z: torch.tensor, p_union: torch.tensor, n_classes: int):
        """
        Computes the loss of UnionNet according to the article [1].

        Parameters:
        -----------
        z : torch.tensor of shape (n_samples, n_annotators)
            Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
            `i`. A missing label is denoted by `c=-1`.
        p_union : torch.tensor of shape (n_samples, n_annotators * n_classes)
            Probabilities referring to the union over the possible annotations of all annotators, where `p_union[i]`
            refers to the probability vector for sample `i`.
        n_classes : int
            Number of classes.
        """
        # Create and concatenate one-hot encoded annotations: cf. Eq. (2) in the article [1].
        z_one_hot = F.one_hot(z + 1, num_classes=n_classes + 1)[:, :, 1:].float()
        z_one_hot = z_one_hot.flatten(start_dim=1, end_dim=2)

        # Compute the cross entropy loss: cf. Eq. (6) in the article [1].
        n_samples = len(z)
        loss = (-z_one_hot * p_union.log()).sum() / n_samples
        return loss

    @torch.no_grad()
    def get_gt_parameters(self, **kwargs):
        gt_parameters = list(self.gt_embed_x.parameters())
        gt_parameters += list(self.gt_output.parameters())
        return gt_parameters

    @torch.no_grad()
    def get_ap_parameters(self, **kwargs):
        ap_parameters = self.transition_matrix
        return ap_parameters
