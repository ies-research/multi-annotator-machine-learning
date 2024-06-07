import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict

from ._base import MaMLClassifier


class CrowdARClassifier(MaMLClassifier):
    """CrowdARClassifier

    CrowdAR (Crowds with Annotation Reliability) [1] is an end-to-end learning solution, which jointly estimates the
    annotators' confusion matrices and annotation reliabilities.

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
    gt_embed_dim_x : int
        Dimension of the sample embeddings learned and outputted by `gt_embed_x`.
    ap_embed_dim_x : int, optional (default=128)
        Dimension of the sample difficulty embedding learned by the reliability network.
    lmbda : float, optional (default=0.9)
        Parameter controlling the convex combination of the annotation and reliability loss terms.
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
    [1] Cao, Z., Chen, E., Huang, Y., Shen, S., & Huang, Z. (2023, July). Learning from Crowds with Annotation
        Reliability. IInternational ACM SIGIR Conference on Research and Development in Information Retrieval
        (pp. 2103-2107).
    """

    def __init__(
        self,
        n_classes: int,
        n_annotators: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        gt_embed_dim_x: int,
        ap_embed_dim_x: int = 128,
        lmbda: float = 0.9,
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
        self.ap_embed_dim_x = ap_embed_dim_x
        self.lmbda = lmbda

        # Set up ground truth (GT) model.
        self.gt_output = gt_output
        self.gt_embed_dim_x = gt_embed_dim_x

        # Set up annotator performance (AP) model.
        self.ap_expertise = nn.Parameter(torch.empty(self.n_annotators, self.n_classes))
        nn.init.xavier_normal_(self.ap_expertise)
        self.ap_confs = nn.Parameter(torch.stack([torch.eye(self.n_classes) for _ in range(self.n_annotators)]))
        self.ap_reliability = ReliabilityNetwork(
            input_dim_x=self.gt_embed_dim_x,
            n_classes=self.n_classes,
            embed_dim_x=self.ap_embed_dim_x,
        )

        # Deactivate automatic optimization to allow training with constraints.
        self.automatic_optimization = False

        # Save hyper parameters.
        self.save_hyperparameters(logger=False)

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
        p_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership probabilities.
        logits_annot : torch.Tensor of shape (batch_size, n_annotators, n_classes)
            Annotation logits for each sample-annotator pair.
        """
        # Compute logits.
        x_embedding = self.gt_embed_x(x)
        logits_class = self.gt_output(x_embedding)

        # Compute class-membership probabilities.
        p_class = F.softmax(logits_class, dim=-1)

        if not return_ap_outputs:
            return p_class

        # Compute annotation reliabilities: cf. Eq. (2) in the article [1].
        p_reliability = self.ap_reliability(x_embedding.detach(), self.ap_expertise, p_class)

        # Compute annotation logits: cf. Eq. (4) in the article [1].
        logits_annot = torch.einsum("ik,jkl->ijl", p_class, self.ap_confs)

        return p_class, logits_annot, p_reliability

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        # Get optimizer.
        optimizer = self.optimizers()

        # Compute losses of the networks.
        p_class, logits_annot, p_reliability = self.forward(x=batch["x"], return_ap_outputs=True)
        loss = self.loss(
            z=batch["z"],
            p_class=p_class,
            logits_annot=logits_annot,
            p_reliability=p_reliability,
            lmbda=self.lmbda,
        )

        # Perform optimization.
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        # Apply constraint to restrict each element of w_3 (cf. Eq. (2) in the article [1]) to be positive.
        self.ap_reliability.apply_clipper()

        # Get learning rate scheduler.
        lr_scheduler = self.lr_schedulers()

        # Perform learning rate scheduler step at the end of each epoch.
        if self.trainer.is_last_batch:
            lr_scheduler.step()

        return loss

    @torch.inference_mode()
    def predict_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        self.eval()
        a = batch.get("a", None)
        if a is None:
            p_class = self.forward(x=batch["x"], return_ap_outputs=False)
            return {"p_class": p_class}
        else:
            p_class, _, p_reliability = self.forward(x=batch["x"], return_ap_outputs=True)
            return {"p_class": p_class, "p_perf": p_reliability}

    @staticmethod
    def loss(
        z: torch.tensor,
        p_class: torch.tensor,
        logits_annot: torch.tensor,
        p_reliability: torch.tensor,
        lmbda: float = 0.9,
    ):
        """
        Computes the loss of CrowdAR according to Eq. (5) in the article [1].

        Parameters:
        -----------
        z : torch.tensor of shape (n_samples, n_annotators)
            Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
            `i`. A missing label is denoted by `c=-1`.
        p_class : torch.tensor of shape (n_samples, n_classes)
            Class-membership probabilities estimated by the GT model.
        logits_annot : torch.tensor of shape (n_samples, n_annotators, n_classes)
            Estimated annotation logits, where `logits_annot[i,j,c]` refers to the estimated logit for sample `i`,
            annotator `j`, and class `c`.
        p_reliability : torch.tensor of shape (n_samples, n_annotators)
            Annotation reliabilities, where `p_reliabilitiy[i,j]` refers to reliability of the annotation provided by
            annotator `j` for sample `i`.
        lmbda : float, optional (default=0.9)
            Parameter controlling the convex combination of the annotation and reliability loss terms.
        """
        # Define number of class and mask of missing annotations.
        n_classes = p_class.shape[-1]
        is_missing = z == -1

        # Compute soft annotation targets: cf. Eq. (3) in the article [1].
        z_one_hot = torch.eye(n_classes, device=z.device)[z]
        z_uniform = torch.full_like(z_one_hot, fill_value=1 / n_classes, device=z.device)
        z_soft = p_reliability[:, :, None] * z_one_hot + (1 - p_reliability[:, :, None]) * z_uniform
        z_soft[is_missing] = 0

        # Compute loss of predicted annotations: cf. first summand of Eq. (5) in the article [1].
        loss_annot = -(z_soft * logits_annot.log_softmax(dim=-1)).sum()

        # Compute targets for the reliability network: cf. indicator function in Eq. (5) in the article [1].
        y_class = p_class.argmax(dim=-1)
        is_correct = (y_class[:, None] == z).float()

        # Compute loss of predicted annotation reliability: cf. right summand of Eq. (5) in the article [1].
        loss_reliability = F.binary_cross_entropy(
            p_reliability, is_correct, weight=(~is_missing).float(), reduction="sum"
        )

        # Sum and normalize losses according to the number of labels: cf. Eq. (5) in the article [1] and its code.
        n_labels = (~is_missing).sum().float().item()
        loss = (lmbda * loss_annot + (1 - lmbda) * loss_reliability) / n_labels

        return loss

    @torch.no_grad()
    def get_gt_parameters(self, **kwargs):
        gt_parameters = list(self.gt_embed_x.parameters())
        gt_parameters += list(self.gt_output.parameters())
        return gt_parameters

    @torch.no_grad()
    def get_ap_parameters(self, **kwargs):
        ap_parameters = list(self.ap_reliability.parameters())
        ap_parameters += list(self.ap_confs)
        ap_parameters += list(self.ap_expertise)
        return ap_parameters


class ReliabilityNetwork(nn.Module):
    """
    ReliabilityNetwork

    A neural network which estimates the reliability of each annotation according to [1].

    Parameters
    ----------
    input_dim_x : int
        Dimension of the samples.
    n_classes : int
        Number of classes.
    embed_dim_x : int, optional (default=128)
        Dimension of the embeddings learning by the sample difficulty layers.

    References
    ----------
    [1] Cao, Z., Chen, E., Huang, Y., Shen, S., & Huang, Z. (2023, July). Learning from Crowds with Annotation
        Reliability. IInternational ACM SIGIR Conference on Research and Development in Information Retrieval
        (pp. 2103-2107).
    """

    def __init__(self, input_dim_x: int, n_classes: int, embed_dim_x: int = 128):
        super().__init__()
        self.w_1 = nn.Linear(input_dim_x, embed_dim_x)
        self.w_2 = nn.Linear(embed_dim_x, n_classes)
        self.w_3 = nn.Linear(n_classes, 1)

    def forward(self, x: torch.tensor, e: torch.tensor, q: torch.tensor):
        """
        Computes the reliability of each annotation.

        Parameters
        ----------
        x : torch.tensor of shape (batch_size, input_dim_x)
            Instance feature vectors
        e : torch.tensor of shape (n_annotators, n_classes)
            Annotator expertise vectors.
        q : torch.tensor of shape (batch_size, n_classes)
            Estimated class-membership probabilities of each sample.

        Returns
        -------
        m : torch.tensor of shape (batch_size, n_annotators)
            Estimated annotation reliabilities.
        """
        # Compute sample difficulty: cf. Eq. (1) in the article [1].
        h = torch.sigmoid(self.w_1(x))
        d = torch.sigmoid(self.w_2(h))

        # Compute annotator expertise: cf. Section 3.2.2 in the article [1].
        e = torch.sigmoid(e)

        # Compute annotation reliability: cf. Eq. (2) in the article [1].
        h = (e[None, :, :] - d[:, None, :]) * q[:, None, :]
        m = torch.sigmoid(self.w_3(h)).squeeze()
        return m

    def apply_clipper(self):
        """
        Clips the weights `w_3` to ensure that each weight is positive.
        """
        w = self.w_3.weight.data
        a = torch.relu(torch.neg(w))
        w.add_(a)
