import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict, Literal

from maml.classifiers._base import MaMLClassifier


class MaskedParameter3D(nn.Module):
    r"""Sparse learnable tensor.

    A compact alternative to a full ``(n_samples, n_annotators, n_classes)`` weight tensor where only a subset
    of *sample–annotator* pairs is trainable.

    The subset is specified via a boolean *mask* of shape ``(n_samples, n_annotators)``.  A ``True`` entry in the mask
    means that the **entire row of classes** for that *⟨sample, annotator⟩* pair is learnable, while ``False`` rows
    are treated as constant zeros and are not stored in memory.

    **Memory footprint** grows with ``mask.sum() * n_classes`` instead of the full product
    ``n_samples * n_annotators * n_classes``.

    Indexing in the *sample* dimension works like plain tensors, e.g.::

        obj[[0, 3]]  # -> (2, n_annotators, n_classes)

    Notes
    -----
    * Forward propagation is free because the module exposes its parameters
      directly; there is no custom ``forward`` method.
    * All helper logic lives in :py:meth:`_dense_rows` and
      :py:meth:`__getitem__`.
    """

    def __init__(self, mask: torch.Tensor, n_classes: int, *, init: str = "zeros"):
        """Create a sparse 3‑D parameter tensor.

        Parameters
        ----------
        mask : torch.Tensor of dtype ``bool``
            Boolean mask ``(n_samples, n_annotators)`` that selects the trainable *⟨sample, annotator⟩* pairs.
        n_classes : int
            Number of classes C (width of each stored row).
        init : {'zeros', 'normal', 'uniform'}, default ``'zeros'``
            Initialisation strategy for the compact parameter block.

            * 'zeros' - all weights are initialised to ``0``.
            * 'normal' - samples from N(0, 0.02)`.
            * 'uniform' - samples from U(-0.05, 0.05)`.

        Raises
        ------
        TypeError
            If *mask* is not a boolean tensor.
        ValueError
            If *init* is not one of the supported strategies.
        """
        super().__init__()

        if mask.dtype is not torch.bool:
            raise TypeError("mask must be a bool tensor")

        self.n_classes = int(n_classes)

        # (n_samples, n_annotators)  →  flat index or −1 sentinel
        index_mat = torch.full(mask.shape, -1, dtype=torch.long)
        true_pos = mask.nonzero(as_tuple=False)  # (K, 2)
        index_mat[true_pos[:, 0], true_pos[:, 1]] = torch.arange(true_pos.size(0), dtype=torch.long)
        self.register_buffer("index_mat", index_mat)

        # compact learnable block
        n_params = true_pos.size(0)
        param = torch.empty(n_params, n_classes)  # (K, C)

        if init == "zeros":
            nn.init.zeros_(param)
        elif init == "normal":
            nn.init.normal_(param, 0.0, 0.02)
        elif init == "uniform":
            nn.init.uniform_(param, -0.05, 0.05)
        else:
            raise ValueError(f"unknown init '{init}'")

        self.param = nn.Parameter(param)  # learnable

    def _dense_rows(self, sample_idx: torch.Tensor) -> torch.Tensor:
        """Return a dense slice for *sample_idx*.

        Parameters
        ----------
        sample_idx : 1‑D ``LongTensor``
            Indices along the *sample* dimension.

        Returns
        -------
        torch.Tensor
            Dense tensor of shape ``(len(sample_idx), n_annotators, n_classes)``
            that contains the requested samples.  Non‑learnable rows are
            zeros; learnable rows are gathered from the compact store.
        """
        idx = self.index_mat[sample_idx]
        out = torch.zeros(
            (*idx.shape, self.n_classes),
            dtype=self.param.dtype,
            device=self.param.device,
        )  # (S, A, C)

        valid = idx >= 0  # bool mask of stored rows
        if valid.any():
            flat = idx[valid]  # (nnz,)
            out[valid] = self.param[flat]  # broadcast row vector
        return out

    def __getitem__(self, item):
        """Index in the *sample* dimension.

        Parameters
        ----------
        item : int | list[int] | torch.LongTensor | slice
            Selection of samples.

        Returns
        -------
        torch.Tensor
            * If *item* is an ``int``: returns a tensor of shape
              ``(n_annotators, n_classes)``.
            * Otherwise: returns ``(len(item), n_annotators, n_classes)``.
        """
        if isinstance(item, slice):
            item = torch.arange(
                *item.indices(self.index_mat.size(0)),
                dtype=torch.long,
                device=self.index_mat.device,
            )
        elif isinstance(item, int):
            return self._dense_rows(torch.tensor([item], device=self.index_mat.device))[0]  # drop batch dim
        elif isinstance(item, (list, tuple)):
            item = torch.as_tensor(item, dtype=torch.long, device=self.index_mat.device)

        return self._dense_rows(item)

    def extra_repr(self) -> str:
        """Human‑readable string for ``print(module)``.

        Returns
        -------
        str
            Formatted description *'sparse_shape=(N, A, C), nnz_elements=K*C'*.
        """
        n_samples, n_ann = self.index_mat.shape
        nnz = self.param.numel()
        return f"sparse_shape=({n_samples}, {n_ann}, {self.n_classes}), " f"nnz_elements={nnz}"


class RegCrowdNetClassifier(MaMLClassifier):
    """
    Regularized crowd network classifier that jointly learns ground‑truth label distributions and per‑annotator
    confusion matrices [1–3].

    This model extends a base MaMLClassifier by introducing a penalty term on each annotator’s confusion matrix, either
    via trace‑regularization or a geometry‑inspired penalty—to encourage the learned matrices to converge to the true
    noise patterns. Optionally, an instance‑dependent outlier penalty (“coin‑net”) can be applied.

    Parameters
    ----------
    n_classes : int
        Number of target classes, C.
    n_annotators : int
        Number of distinct annotators, A.
    gt_embed_x : torch.nn.Module
        Backbone embedding network for the ground‑truth model. Maps raw inputs
        to a feature embedding.
    gt_output : torch.nn.Module
        Head network for the ground‑truth model. Maps the embedding to
        class‑membership logits.
    n_samples : int, optional (default=-1)
        Total number of training samples, N. Only required if using
        instance‑dependent regularization; set to –1 to infer dynamically.
    lmbda : float >= 0, optional (default=0.01)
        Regularization weight on annotator confusion matrices.
    mu : float >= 0, optional (default=0.01)
        Instance‑outlier penalty weight (only used with “coin‑net” mode).
    p : float in (0, 1], optional (default=0.4)
        p‑norm exponent for the instance‑outlier penalty (coin‑net only).
    regularization : {'trace-reg', 'geo-reg-f', 'geo-reg-w', 'coin-net'}, optional
        Choice of confusion‑matrix regularizer:
        - 'trace-reg': trace penalty on each matrix [1]
        - 'geo-reg-f': Frobenius‑geodesic penalty [2]
        - 'geo-reg-w': Wasserstein‑geodesic penalty [2]
        - 'coin-net': Frobenius‑geodesic penalty with instance-dependent outliers [3]
    annotation_mask : torch.Tensor of shape (N, A), optional
        Boolean mask indicating which ⟨sample, annotator⟩ pairs are observed. Missing entries are ignored during loss
        computation (coin‑net only).
    optimizer : torch.optim.Optimizer class, optional
        Optimizer for both GT and annotator parameters (default: RAdam).
    optimizer_gt_dict : dict, optional
        Keyword arguments passed to `optimizer` for the GT model.
    optimizer_ap_dict : dict, optional
        Keyword arguments passed to `optimizer` for the annotator parameters.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler class, optional
        Learning‑rate scheduler for both GT and annotator optimizers.
    lr_scheduler_dict : dict, optional
        Keyword arguments passed to `lr_scheduler`.

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
        lmbda: float = 0.01,
        mu: float = 0.01,
        p: float = 0.4,
        regularization: Literal["trace-reg", "geo-reg-f", "geo-reg-w"] = "trace-reg",
        annotation_mask: Optional[torch.Tensor] = None,
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
        self.mu = mu
        self.p = p
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
            if annotation_mask is None:
                annotation_mask = torch.full((self.n_samples, self.n_annotators, self.n_classes), fill_value=True)
            self.ap_outlier_terms = MaskedParameter3D(mask=annotation_mask.bool(), n_classes=self.n_classes)

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
            ap_outlier_terms=self.ap_outlier_terms[batch["idx"]] if self.ap_outlier_terms is not None else None,
            lmbda=self.lmbda,
            mu=self.mu,
            p=self.p,
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
            p_confusion_log = F.log_softmax(self.ap_confs[a], dim=-1)
            p_confusion = (p_class_log[:, None, :, None] + p_confusion_log).exp()
            p_perf = p_confusion.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            p_annot = torch.logsumexp(p_class_log[:, None, :, None] + p_confusion_log, dim=2).exp()
            return {
                "p_class": p_class_log.exp(),
                "p_perf": p_perf,
                "p_conf": p_confusion_log.exp(),
                "p_annot": p_annot,
            }

    @staticmethod
    def loss(
        z: torch.tensor,
        logits_class: torch.tensor,
        ap_confs: torch.tensor,
        ap_outlier_terms: Optional[torch.tensor] = None,
        lmbda: float = 0.01,
        mu: float = 0.01,
        p: float = 0.4,
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
            Regularization term penalizing confusion matrices.
        mu : non-negative float, optional (default=0.01)
            Regularization term penalizing instance-dependent outliers. Only relevant for "coin-net".
        p : non-negative float in (0, 1], optional (default=0.4)
            Norm to compute regularization term for instance-dependent outliers. Only relevant for "coin-net".
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
        e_outlier_err = 0
        if regularization == "coin-net":
            e_outlier = ap_outlier_terms - ap_outlier_terms.mean(dim=-1, keepdim=True)
            e_outlier_err = (((e_outlier**2).sum(dim=(1, 2)) + 1e-10) ** (0.5 * p)).sum()
            e_outlier = e_outlier.reshape(-1, e_outlier.shape[-1])[is_lbld]
            p_annot = p_annot_log.exp() + e_outlier
            p_annot = p_annot.clamp(min=1e-10, max=1 - 1e-10)
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
            elif regularization == "geo-reg-w":
                p_perf = p_perf_log.exp().swapaxes(1, 2).flatten(start_dim=0, end_dim=1)
                # Cf. second summand of Eq. (9) in [2].
                reg_term = -torch.logdet(p_perf.T @ p_perf)
                # Cf. proposed code in the GitHub repository of [2].
                if torch.isnan(reg_term) or torch.isinf(torch.abs(reg_term)) or reg_term > 100:
                    reg_term = 0
            else:
                raise ValueError("`regularization` must be in ['trace-reg', 'geo-reg-f', 'geo-reg-w'].")
            loss += lmbda * reg_term + mu * e_outlier_err
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
