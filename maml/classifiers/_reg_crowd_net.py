import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict, Literal, Union

from maml.classifiers._base import MaMLClassifier


class RegCrowdNetClassifier(MaMLClassifier):
    """RegCrowdNetClassifier

    The "regularized crowd network" (RegCrowdNet) [1, 2, 3] jointly learns the underlying ground truth (GT)
    distribution and the individual confusion matrix as proxy of each annotator's performance. Therefor, a
    regularization term is added to the loss function that encourages convergence to the true annotator confusion
    matrix.

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
    lmbda : non-negative float, optional (default=0.01)
        Degree of regularization.
    regularization : "trace-reg" or "geo-reg-f" or "geo-reg-w"
        Defines which regularization for the annotator confusion matrices is applied, either by regularizing the traces
        of the confusion matrices [1] or a geometrically motivated regularization [2].
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
    [1] Tanno, Ryutaro, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C. Alexander, and Nathan Silberman.
        "Learning from noisy labels by regularized estimation of annotator confusion." IEEE/CVF Conf. Comput. Vis.
         Pattern Recognit., pp. 11244-11253. 2019.
    [2] Ibrahim, Shahana, Tri Nguyen, and Xiao Fu. "Deep Learning From Crowdsourced Labels: Coupled Cross-Entropy
        Minimization, Identifiability, and Regularization." Int. Conf. Learn. Represent. 2023.
    [3] Tri Nguyen, Ibrahim, Shahana, and Xiao Fu. "Noisy Label Learning with Instance-Dependent Outliers:
        Identifiability via Crowd Wisdom." Adv. Neural Inf. Process. Syst. 2024.
    """

    def __init__(
        self,
        n_classes: int,
        n_annotators: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        n_samples: Optional[int] = -1,
        lmbda: Union[str, float] = "auto",
        regularization: Literal["trace-reg", "geo-reg-f", "geo-reg-w"] = "trace-reg",
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
        self.n_samples = n_samples
        self.lmbda = lmbda
        self.regularization = regularization

        # Perform initialization of confusion matrices and potential outlier terms.
        if regularization == "trace-reg":
            # Cf. code snippet in Appendix of [1] for proposed initialization.
            self.ap_confs = nn.Parameter(torch.stack([6.0 * torch.eye(n_classes) - 5.0] * n_annotators))
        elif regularization in ["geo-reg-f", "geo-reg-w", "coin-net"]:
            # Cf. Section 5 in [2] or Appendix G.1 in [3] for proposed initialization.
            self.ap_confs = nn.Parameter(torch.stack([torch.eye(n_classes)] * n_annotators))
        else:
            raise ValueError("`regularization` must be in ['trace-reg', 'geo-reg-f', 'geo-reg-w'].")
        self.ap_outlier_terms = None
        if regularization == "coin-net":
            # Cf. Appendix G.1 in [3] for proposed initialization.
            self.ap_outlier_terms = nn.Parameter(torch.zeros((self.n_samples, self.n_annotators, self.n_classes)))

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
        # Compute logits.
        x_learned = self.gt_embed_x(x)
        logits_class = self.gt_output(x_learned)
        return logits_class

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        """
        Computes the RegCrowdNet's loss.

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
        logits_class = self.forward(x=batch["x"])
        loss = RegCrowdNetClassifier.loss(
            z=batch["z"],
            logits_class=logits_class,
            ap_confs=self.ap_confs,
            ap_outlier_terms=self.ap_outlier_terms[batch["idx"]],
            lmbda=self.lmbda,
            regularization=self.regularization,
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
        output = self.forward(x=batch["x"])
        if a is None:
            return {"p_class": F.softmax(output, dim=-1)}
        else:
            p_class_log = F.log_softmax(output, dim=-1)
            p_confusion_log = F.log_softmax(self.ap_confs[batch["a"]], dim=-1)
            p_confusion = (p_class_log[:, None, :, None] + p_confusion_log).exp()
            p_perf = p_confusion.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            return {"p_class": p_class_log.exp(), "p_perf": p_perf}

    @staticmethod
    def loss(
        z: torch.tensor,
        logits_class: torch.tensor,
        ap_confs: torch.tensor,
        ap_outlier_terms: Optional[torch.tensor] = None,
        lmbda: float = 0.01,
        regularization: Literal["trace-reg", "geo-reg-f", "geo-reg-w"] = "trace-reg",
    ):
        """
        Computes RegCrowdNet's loss according either to the article [1] or to the article [2] or to the article [3].

        Parameters
        ----------
        z : torch.tensor of shape (n_samples, n_annotators)
            Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
            `i`. A missing label is denoted by `c=-1`.
        logits_class : torch.tensor of shape (n_samples,, n_classes)
            Estimated class-membership logits, where `logits_class[i,c]` refers to the estimated logit for sample `i`,
            and class `c`.
        ap_confs : torch.tensor of shape (n_annotators, n_classes, n_classes)
            Annotator-dependent confusion matrices, where `ap_confs[j]` refers to the confusion matrix of
            annotator `j`.
        ap_outlier_terms: torch.tensor of shape (n_samples, n_annotators, n_classes) or None
            This is an optional parameter, which is only relevant if `regularization="coin-net"` [3].
        lmbda : non-negative float, optional (default=0.01)
            Regularization term penalizing the sums of diagonals of annotators' confusion matrices.
        regularization : "trace-reg" or "geo-reg-f" or "geo-reg-w" or "coin-net"
            Defines which regularization for the annotator confusion matrices is applied, either by regularizing the
            traces of the confusion matrices [1], a geometrically motivated regularization [2], or an additional
            instance- and annotator-dependent regularization term.

        References
        ----------
        [1] Tanno, Ryutaro, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C. Alexander, and Nathan Silberman.
            "Learning from noisy labels by regularized estimation of annotator confusion." IEEE/CVF Conf. Comput. Vis.
             Pattern Recognit., pp. 11244-11253. 2019.
        [2] Ibrahim, Shahana, Tri Nguyen, and Xiao Fu. "Deep Learning From Crowdsourced Labels: Coupled Cross-Entropy
            Minimization, Identifiability, and Regularization." Int. Conf. Learn. Represent. 2023.
        [3] Tri Nguyen, Ibrahim, Shahana, and Xiao Fu. "Noisy Label Learning with Instance-Dependent Outliers:
            Identifiability via Crowd Wisdom." Adv. Neural Inf. Process. Syst. 2024.
        """
        n_samples, n_annotators = z.shape[0], z.shape[1]
        combs = torch.cartesian_prod(
            torch.arange(n_samples, device=z.device), torch.arange(n_annotators, device=z.device)
        )
        z = z.ravel()
        is_lbld = z != -1
        combs, z = combs[is_lbld], z[is_lbld]

        # Compute log-probabilities for the annotations.
        p_class_log = F.log_softmax(logits_class, dim=-1)
        p_class_log_ext = p_class_log[combs[:, 0]]
        p_perf_log = torch.log_softmax(ap_confs, dim=-1)
        p_perf_log_ext = p_perf_log[combs[:, 1]]
        p_annot_log = torch.logsumexp(p_class_log_ext[:, :, None] + p_perf_log_ext, dim=1)

        # Incorporate outlier terms into the predicted annotation probabilities.
        e_outlier = None
        if regularization == "coin-net":
            e_outlier = ap_outlier_terms - ap_outlier_terms.mean(dim=-1, keepdim=True)
            e_outlier = e_outlier.reshape(-1, e_outlier.shape[-1])[is_lbld]
            p_annot = p_annot_log.exp() + e_outlier
            p_annot = p_annot.clamp(min=1e-10, max=1-1e-10)
            p_annot = p_annot / p_annot.sum(-1, keepdim=True)
            p_annot_log = p_annot.log()

        # Compute cross-entropy term.
        loss = F.nll_loss(p_annot_log, z, reduction="mean", ignore_index=-1)

        # Compute and add regularization terms.
        if lmbda > 0:
            if regularization == "trace-reg":
                # Cf. second summand of Eq. (4) in [1].
                p_perf = F.softmax(ap_confs, dim=-1)
                reg_term = p_perf.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).mean()
            elif regularization in ["geo-reg-f", "coin-net"]:
                p_class = p_class_log.exp()
                # Cf. second summand of Eq. (8) in [2].
                reg_term = -torch.logdet(p_class.T @ p_class)
                # Cf. proposed code in the GitHub repository of [2].
                if torch.isnan(reg_term) or torch.isinf(torch.abs(reg_term)) or reg_term > 100:
                    reg_term = 0
                if regularization == "coin-net":
                    err = (e_outlier ** 2).sum((1, 2)) + 1e-10
                    reg_term = reg_term + ((err ** 0.2).mean())
            elif regularization == "geo-reg-w":
                p_perf = p_perf_log.exp().swapaxes(1, 2).flatten(start_dim=0, end_dim=1)
                # Cf. second summand of Eq. (9) in [2].
                reg_term = -torch.logdet(p_perf.T @ p_perf)
                # Cf. proposed code in the GitHub repository of [2].
                if torch.isnan(reg_term) or torch.isinf(torch.abs(reg_term)) or reg_term > 100:
                    reg_term = 0
            else:
                raise ValueError("`regularization` must be in ['trace-reg', 'geo-reg-f', 'geo-reg-w'].")
            loss += lmbda * reg_term
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
        ap_parameters = self.ap_confs
        return ap_parameters
