import torch


from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict

from ._base import MaMLClassifier


class CoNALClassifier(MaMLClassifier):
    """CoNALClassifier

    CoNAL (Common Noise Adaption Layers) [1] is an end-to-end learning solution with two types of noise adaptation
    layers: one is shared across annotators to capture their commonly shared confusions, and the other one is
    pertaining to each annotator to realize individual confusion. The implementation is based on the GitHub
    repository [2].

    Parameters
    ----------
    n_classes : int
        Number of classes.
    annotators : torch.tensor of shape (n_annotators, n_annotator_features)
        The matrix `A` is the annotator feature matrix. The exact form depends on the `module_class`.
        If it is `None`, a one-hot encoding is used to differentiate between annotators.
    gt_embed_x : nn.Module
        Pytorch module the GT model' backbone embedding the input samples.
    gt_output : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
    ap_embed_a : nn.Module
        Pytorch module of the AP model embedding the annotator features for the AP model.
    ap_embed_x : nn.Module, optional (default=None)
        Pytorch module of the AP model embedding samples.
    lmbda : float, optional (default=1e-5)
        Regularization parameter to enforce the common and individual confusion matrices of the annotators to be
        different.
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
    [1] Chu, Z., Ma, J., & Wang, H. (2021, May). Learning from Crowds by Modeling Common Confusions.
        In AAAI Int. Conf. Artif. Intell. (pp. 5832-5840).
    [2] Chu, Z (GitHub username: zdchu). GitHub Repository: https://github.com/zdchu/CoNAL.
    """

    def __init__(
        self,
        n_classes: int,
        annotators: torch.tensor,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        ap_embed_a: nn.Module,
        ap_embed_x: nn.Module,
        ap_use_gt_embed_x: bool = True,
        lmbda: float = 1e-5,
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
        self.register_buffer("annotators", annotators)
        self.gt_embed_x = gt_embed_x
        self.gt_output = gt_output
        self.ap_embed_a = ap_embed_a
        self.ap_embed_x = ap_embed_x
        self.ap_use_gt_embed_x = ap_use_gt_embed_x
        self.lmbda = lmbda
        self.ap_confs_individual = nn.Parameter(
            torch.stack([2 * torch.eye(n_classes)] * len(self.annotators)),
        )
        self.ap_confs_common = nn.Parameter(2.0 * torch.eye(n_classes).float())
        self.save_hyperparameters()

    def forward(self, x: torch.tensor, return_ap_outputs: bool = False):
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.
        return_ap_outputs: bool, optional (default=True)
            Flag whether the annotation logits are to be returned, next to the class-membership probabilities.

        Returns
        -------
        p_class : torch.tensor of shape (batch_size, n_classes)
            Class-membership probabilities.
        logits_annot : torch.tensor of shape (batch_size, n_annotators, n_classes)
            Annotation logits for each sample-annotator pair.
        """
        # Compute logits.
        x_embedding = self.gt_embed_x(x)
        logits_class = self.gt_output(x_embedding)

        # Compute class-membership probabilities.
        p_class = F.softmax(logits_class, dim=-1)

        if not return_ap_outputs:
            return p_class

        # Compute embeddings and normalize them.
        if self.ap_use_gt_embed_x:
            x_embedding = self.ap_embed_x(x_embedding)
        else:
            x_embedding = self.ap_embed_x(x)
        x_embedding = F.normalize(x_embedding)
        a_embedding = F.normalize(self.ap_embed_a(self.annotators))

        # Take product of embeddings to compute probability of common confusion matrix: cf. Eq. (3) in the article [1].
        common_rate = torch.einsum("ij,kj->ik", (x_embedding, a_embedding))
        common_rate = F.sigmoid(common_rate)

        # Compute common confusion matrix and individual confusion matrix products.
        logits_common = torch.einsum("ij,jk->ik", (p_class, self.ap_confs_common))
        logits_individual = torch.einsum("ik,jkl->ijl", (p_class, self.ap_confs_individual))

        # Compute logits per annotator as pre-step of computing the final probability distribution of annotations:
        # cf. lower part of Eq. (1) in the article [1].
        logits_annot = common_rate[:, :, None] * logits_common[:, None, :]
        logits_annot += (1 - common_rate[:, :, None]) * logits_individual
        logits_annot = logits_annot

        return p_class, logits_annot

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        """
        Computes CoNAL's loss.

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
            Computed cross-entropy loss with regularization.
        """
        _, logits_annot = self.forward(x=batch["x"], return_ap_outputs=True)
        loss = CoNALClassifier.loss(
            z=batch["z"],
            logits_annot=logits_annot,
            ap_confs_individual=self.ap_confs_individual,
            ap_confs_common=self.ap_confs_common,
            lmbda=self.lmbda,
        )
        return loss

    @torch.inference_mode()
    def predict_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
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
            p_annot = F.softmax(logits_annot, dim=-1)
            p_perf = torch.stack([torch.einsum("ij,ik->ijk", p_class, p_annot[:, i, :]) for i in range(a.shape[-1])])
            p_perf = p_perf.swapaxes(0, 1).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            return {"p_class": p_class, "p_perf": p_perf}

    @staticmethod
    def loss(
        z: torch.tensor,
        logits_annot: torch.tensor,
        ap_confs_individual: torch.tensor,
        ap_confs_common: torch.tensor,
        lmbda: float = 1e-5,
    ):
        """
        Computes the loss of CoNAL according to the article [1].

        Parameters:
        -----------
        z : torch.tensor of shape (n_samples, n_annotators)
            Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
            `i`. A missing label is denoted by `c=-1`.
        logits_annot : torch.tensor of shape (n_samples, n_annotators, n_classes)
            Estimated annotation logits, where `logits_annot[i,j,c]` refers to the estimated logit for sample `i`,
            annotator `j`, and class `c`.
        ap_confs_individual : torch.tensor of shape (n_annotators, n_classes, n_classes)
            Annotator-dependent confusion matrices, where `ap_confs_individual[j]` refers to the confusion matrix of
            annotator `j`.
        ap_confs_common : torch.tensor of shape (n_classes, n_classes)
            Common confusion matrix.
        lmbda : float, optional (default=1e-5)
            Parameter controlling the importance of the difference between the common confusion matrix and individual
            confusion matrices.

        References
        ----------
        [1] Chu, Z., Ma, J., & Wang, H. (2021, May). Learning from Crowds by Modeling Common Confusions.
        In AAAI Int. Conf. Artif. Intell. (pp. 5832-5840).
        """
        # Compute cross entropy for annotation probabilities according to the equation of final loss function in the
        # article [1].
        loss = F.cross_entropy(logits_annot.swapaxes(1, 2), z, reduction="mean", ignore_index=-1)
        if lmbda > 0:
            # Compute regularization term according to the equation of final loss function in the article [1].
            diff = (ap_confs_individual - ap_confs_common).view(z.shape[1], -1)
            norm_sum = diff.norm(dim=1, p=2).sum()
            loss -= lmbda * norm_sum
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
        ap_parameters = list(self.ap_embed_a.parameters())
        ap_parameters += list(self.ap_embed_x.parameters())
        ap_parameters += list(self.ap_confs_common)
        ap_parameters += list(self.ap_confs_individual)
        return ap_parameters
