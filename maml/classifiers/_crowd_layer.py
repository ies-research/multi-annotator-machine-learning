import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict

from maml.classifiers._base import MaMLClassifier


class CrowdLayerClassifier(MaMLClassifier):
    """CrowdLayerClassifier

    CrowdLayer [1] is a layer added at the end of a classifying neural network and allows us to train deep neural
    networks end-to-end, directly from the noisy labels of multiple annotators, using only backpropagation.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    n_annotators : int
        Number of annotators.
    gt_embed_x : nn.Module
        Pytorch module the GT model' backbone embedding the input samples.
    gt_output : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
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

    References
    ----------
    [1] Rodrigues, F., & Pereira, F. Deep Learning from Crowds. AAAI Conf. Artif. Intell.
    """

    def __init__(
        self,
        n_classes: int,
        n_annotators: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        optimizer: Optional[Optimizer.__class__] = RAdam,
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

        # Initialize crowd layer with identity matrices: cf. MV approach in the article [1].
        self.ap_crowd_layer = nn.Parameter(torch.stack([torch.eye(n_classes) for _ in range(n_annotators)], dim=2))

        # Store hyperparameters to reload weights.
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.tensor, return_ap_outputs: bool = False):
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.
        return_ap_outputs : bool, optional (default=True)
            Flag whether the annotation logits are to be returned, next to the class-membership probabilities.

        Returns
        -------
        p_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership probabilities.
        logits_annot : torch.Tensor of shape (batch_size, n_annotators, n_classes)
            Annotation logits for each sample-annotator pair.
        """
        # Compute feature embedding for classifier.
        x_learned = self.gt_embed_x(x)

        # Compute class-membership probabilities.
        logits_class = self.gt_output(x_learned)

        # Compute class-membership probabilities.
        p_class = F.softmax(logits_class, dim=-1)

        if not return_ap_outputs:
            return p_class

        # Compute logits per annotator.
        logits_annot = torch.einsum("nc,ckm->nmk", p_class, self.ap_crowd_layer)

        return p_class, logits_annot

    def training_step(
        self,
        batch: Dict[str, torch.tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """
        Computes the CrowdLayer's loss.

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
        _, logits_annot = self.forward(x=batch["x"], return_ap_outputs=True)
        loss = CrowdLayerClassifier.loss(
            z=batch["z"],
            logits_annot=logits_annot,
        )
        return loss

    @torch.inference_mode()
    def predict_step(
        self,
        batch: Dict[str, torch.tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        """
        Computes the GT and (optionally) AP models' predictions.

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
        self.eval()
        a = batch.get("a", None)
        if a is None:
            p_class = self.forward(x=batch["x"], return_ap_outputs=False)
            return {"p_class": p_class}
        else:
            p_class, logits_annot = self.forward(x=batch["x"], return_ap_outputs=True)
            p_annot = logits_annot.softmax(dim=-1)
            p_perf = torch.stack([torch.einsum("ij,ik->ijk", p_class, p_annot[:, i, :]) for i in a[0]])
            p_perf = p_perf.swapaxes(0, 1).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            return {"p_class": p_class, "p_perf": p_perf}

    @staticmethod
    def loss(
        z: torch.tensor,
        logits_annot: torch.tensor,
    ):
        """
        Computes the cross-entropy loss between the observed annotations and the predicted annotation probabilities
        according to the article [1].

        Parameters:
        -----------
        z : torch.tensor of shape (n_samples, n_annotators)
            Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
            `i`. A missing label is denoted by `c=-1`.
        logits_annot : torch.tensor of shape (n_samples, n_annotators, n_classes)
            Estimated annotation logits, where `logits_annot[i,j,c]` refers to the estimated logit for sample `i`,
            annotator `j`, and class `c`.

        Returns
        -------
        loss : torch.Float
            Computed cross-entropy loss.

        References
        ----------
        [1] Rodrigues, F., & Pereira, F. Deep Learning from Crowds. AAAI Conf. Artif. Intell.
        """
        loss = F.cross_entropy(logits_annot.swapaxes(1, 2), z, reduction="mean", ignore_index=-1)
        return loss

    @torch.no_grad()
    def get_gt_parameters(self, **kwargs):
        """
        Returns the list of parameters of the GT model.

        Returns
        -------
        gt_parameters : list
            The list of the GT models' parameters.
        """
        gt_parameters = list(self.gt_embed_x.parameters())
        gt_parameters += list(self.gt_output.parameters())
        return gt_parameters

    @torch.no_grad()
    def get_ap_parameters(self, **kwargs):
        """
        Returns the list of parameters of the AP model.

        Returns
        -------
        ap_parameters : list
            The list of the AP models' parameters.
        """
        ap_parameters = self.ap_crowd_layer
        return ap_parameters
