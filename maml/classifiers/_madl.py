import math
import torch

from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import LRScheduler
from torch.distributions import Gamma
from torch import nn
from torch.nn import functional as F

from typing import Optional, Literal, Dict, Union

from ._base import MaMLClassifier
from ..utils import cosine_kernel, rbf_kernel


class MaDLClassifier(MaMLClassifier):
    """MaDLClassifier

    This class implements the framework "multi-annotator deep learning" (MaDL) [1], which jointly trains a ground truth
    (GT) model for classification and an annotator performance (AP) model in an end-to-end approach.

    Parameters
    ----------
    n_classes : int
        Number of classes
    gt_embed_x : nn.Module
        Pytorch module the GT model' backbone embedding the input samples.
    gt_output : nn.Module
        Pytorch module of the GT model taking the embedding the samples as input to predict class-membership logits.
    ap_embed_a : nn.Module
        Pytorch module of the AP model embedding the annotator features for the AP model.
    ap_output : nn.Module
        Pytorch module of the AP model predicting the logits of the conditional confusion matrix
    ap_embed_x : nn.Module, optional (default=None)
        Pytorch module of the AP model embedding samples.
    ap_outer_product : nn.Module, optional (default=None)
        Outer product-based layer to model interactions between annotator and sample embeddings. By default, it is None
        and therefore not used.
    ap_hidden : nn.Module, optional (default=None)
        Pytorch module of the AP model taking the concatenation of annotator, sample (optional) and outer product
        (optional) embedding as input to create a new embedding as input to the `ap_output` module.
        By default, it is an identity mapping.
    ap_use_gt_embed_x : bool, optional (default=True)
        Flag whether the learned sample embeddings or the raw samples are used as inputs to `ap_embed_x`. By default,
        it is True and only relevant, if `ap_embed_x` is not None.
    ap_use_residual : bool, optional (default=True)
        Flag whether a residual block is to be applied to the output of the `ap_hidden`. By default, it is True
        and only relevant, if `ap_hidden` is not None.
    eta : float in (0, 1), optional (default=0.8)
        Prior annotator performance, i.e., the probability of obtaining a correct annotation from an arbitrary
        annotator for an arbitrary sample of an arbitrary class.
    confusion_matrix : {'isotropic', 'diagonal', 'full'}
        Determines the type of estimated confusion matrix, where we differ between:
            - 'isotropic' corresponding to a scalar (one output neuron for `ap_output`) from which the confusion matrix
              is constructed,
            - 'diagonal' corresponding to a vector as diagonal (`n_classes` output neurons for `ap_output`) from which
              the confusion matrix is constructed,
            - 'full' corresponding to a vector (`n_classes * n_classes` output neurons for `ap_output`) from which
              the confusion matrix is constructed.
    kernel : "rbf" or "cosine"
        Name of the kernel function to be used.
    alpha : positive float, optional (default=1.25)
        First parameter of the Gamma distribution to regularize the value of `gamma` as bandwidth parameter of the
        radial basis function or cosine kernel. If it is None, no annotator weights are computed.
    beta : positive float, optional (default=1.25)
        Second parameter of the Gamma distribution to regularize the value of `gamma` as bandwidth parameter of the
        radial basis function or cosine kernel. If it is None, no annotator weights are computed.
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
    verbose : bool, optional (default=False)
        Flag whether the learned annotator weights and `gamma` parameter are to be reported after each training epoch.

    References
    ----------
    [1] Herde, Marek, Huseljic, Denis, and Sick, Bernhard. "Multi-annotator Deep Learning: A Modular Probabilistic
        Framework for Classification." Trans. Mach. Learn. Res., 2023.
    """

    def __init__(
        self,
        n_classes: int,
        gt_embed_x: nn.Module,
        gt_output: nn.Module,
        ap_embed_a: nn.Module,
        ap_output: nn.Module,
        ap_embed_x: nn.Module = None,
        ap_outer_product: nn.Module = None,
        ap_hidden: nn.Module = None,
        ap_use_gt_embed_x: bool = True,
        ap_use_residual: bool = True,
        eta: float = 0.8,
        confusion_matrix: Literal["scalar", "diagonal", "full"] = "full",
        kernel: Literal["rbf", "cosine"] = "rbf",
        alpha: Optional[float] = 1.25,
        beta: Optional[float] = 0.25,
        optimizer: Optional[Optimizer.__class__] = RAdam,
        optimizer_gt_dict: Optional[dict] = None,
        optimizer_ap_dict: Optional[dict] = None,
        lr_scheduler: Optional[LRScheduler.__class__] = None,
        lr_scheduler_dict: Optional[dict] = None,
        verbose: Optional[bool] = False,
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_gt_dict=optimizer_gt_dict,
            optimizer_ap_dict=optimizer_ap_dict,
            lr_scheduler=lr_scheduler,
            lr_scheduler_dict=lr_scheduler_dict,
        )
        self.n_classes = n_classes
        self.gt_embed_x = gt_embed_x
        self.gt_output = gt_output
        self.ap_embed_a = ap_embed_a
        self.ap_output = ap_output
        self.ap_hidden = ap_hidden
        self.ap_embed_x = ap_embed_x
        self.ap_use_gt_embed_x = ap_use_gt_embed_x
        self.ap_use_residual = ap_use_residual
        self.ap_outer_product = ap_outer_product
        self.eta = eta
        self.confusion_matrix = confusion_matrix
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta

        self.verbose = verbose
        self.register_buffer("eye", torch.eye(self.n_classes))
        self.register_buffer("one_minus_eye", 1 - self.eye)

        # Set prior distribution for bandwidth parameter.
        self.gamma_dist = None
        self.gamma = None
        if self.alpha is not None or self.beta is not None:
            self.gamma_dist = Gamma(self.alpha, self.beta)
            gamma_mode = max((0, (self.alpha - 1) / self.beta))
            self.gamma = nn.Parameter(torch.tensor(gamma_mode).float())

        # Set prior eta as bias: cf. Eq. (19) in the article [1].
        with torch.no_grad():
            if confusion_matrix in ["scalar", "diagonal"]:
                bias = torch.ones_like(self.ap_output.bias) * (-math.log(1 / self.eta - 1))
            elif confusion_matrix == "full":
                bias = math.log(self.eta * (self.n_classes - 1) / (1 - self.eta)) * self.eye
                bias = bias.flatten()
            self.ap_output.bias = torch.nn.Parameter(bias)

        # Save hyper parameters.
        self.save_hyperparameters(logger=False)

        # Variable to save annotator weights of last forward propagation.
        self.annot_weights = None

    def forward(
        self, x: torch.tensor, a: Optional[torch.tensor] = None, combs: Union[str, None, torch.tensor] = "full"
    ):
        """
        Forward propagation of samples' and annotators' (optional) features through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.tensor of shape (batch_size, *)
            Sample features.
        a : torch.tensor of shape (n_annotators, *), optional (default=None)
            Annotator features, which are None by default. In this case, only the samples are forward propagated
            through the GT model.
        combs : torch.tensor of shape (n_combs, 2), default=None
            If provided, this tensor determines the pairs of samples (indexed by `combs[:, 0]`) and annotators
            (indexed by `combs[:, 1]`) to be propagated through the AP model.

        Returns
        -------
        logits_class : torch.tensor of shape (batch_size, n_classes)
            Class-membership logits.
        ap_confs : torch.tensor of shape (batch_size, n_annotators, n_classes, n_classes)
            Logits of conditional confusion matrices as proxies of the annotators' performances.
        """
        # Compute feature embedding for classifier.
        x_learned = self.gt_embed_x(x)

        # Compute class-membership probabilities.
        logits_class = self.gt_output(x_learned)

        if a is None:
            return logits_class

        # Compute annotator embeddings.
        annot_embeddings = self.ap_embed_a(a)

        # Optionally: Compute sample embeddings.
        if self.ap_embed_x is not None:
            if self.ap_use_gt_embed_x:
                sample_embeddings = self.ap_embed_x(x_learned)
            else:
                sample_embeddings = self.ap_embed_x(x)
        else:
            sample_embeddings = None

        # Combine annotator and sample embeddings.
        if combs == "full":
            n_annotators = a.shape[0]
            n_samples = x.shape[0]
            combs = torch.cartesian_prod(torch.arange(n_samples), torch.arange(n_annotators))
            combs = {"x": combs[:, 0], "a": combs[:, 1]}
        if isinstance(combs, dict) and "a" in combs:
            annot_embeddings = annot_embeddings[combs["a"]]
        embeddings = [annot_embeddings]
        if isinstance(combs, dict) and "x" in combs and sample_embeddings is not None:
            embeddings.append(sample_embeddings[combs["x"]])

        # Optionally compute product of annotator and sample embeddings.
        if self.ap_outer_product is not None:
            embeddings.append(self.ap_outer_product(torch.stack(embeddings, dim=1)))
        embeddings = torch.concat(embeddings, dim=-1)

        # Propagate embeddings through hidden layers.
        if self.ap_hidden is not None:
            embeddings = self.ap_hidden(embeddings)
            # Optionally: Add annotator embeddings as residuals.
            if self.ap_use_residual:
                embeddings = F.relu(embeddings + annot_embeddings)

        # Compute logits of annotator performances.
        logits_perf = self.ap_output(embeddings)

        # Transform logits of annotator performances into confusion matrix.
        if self.confusion_matrix in ["scalar", "diagonal"]:
            logits_perf = self.eye[None] * logits_perf[:, :, None] + self.one_minus_eye[None] * (
                -math.log(self.n_classes - 1)
            )
        elif self.confusion_matrix == "full":
            logits_perf = logits_perf.reshape((-1, self.n_classes, self.n_classes))

        # Optionally: Compute similarities between annotator embeddings as weights for the loss function: cf.
        # Eq. (21)-(23) in the article [1].
        if self.training and self.gamma is not None and self.gamma_dist is not None:
            annot_embed = self.ap_embed_a(a).detach()
            if self.kernel == "rbf":
                sims = rbf_kernel(annot_embed, gamma=torch.clamp(self.gamma, min=1e-5))
            elif self.kernel == "cosine":
                sims = cosine_kernel(annot_embed, gamma=torch.clamp(self.gamma, min=1e-5))
            else:
                raise ValueError(f"`kernel={self.kernel} is invalid.")
            inv_annot_sums = (sims.sum(0)) ** (-1)
            self.annot_weights = inv_annot_sums / inv_annot_sums.sum()
            self.annot_weights *= len(inv_annot_sums)

        return logits_class, logits_perf

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: Optional[int] = 0):
        """
        Computes MaDL's loss.

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
            Computed cross-entropy loss with regularization and annotator weights.
        """
        # Get data.
        x, a, z = batch["x"], batch["a"], batch["z"]
        n_samples, n_annotators = x.shape[0], a.shape[1]

        # Compute all combinations of samples and annotators.
        combs = torch.cartesian_prod(
            torch.arange(len(x), device=self.device), torch.arange(n_annotators, device=self.device)
        )

        # Prepare samples, annotators, and labels for training.
        z, a = z.ravel(), a[0]
        is_lbld = z != -1
        combs, z = combs[is_lbld], z[is_lbld]
        combs = {"x": combs[:, 0], "a": combs[:, 1]}

        logits_class, logits_perf = self.forward(x=x, a=a, combs=combs)
        loss = self.loss(
            z=z,
            logits_class=logits_class[combs["x"]],
            logits_perf=logits_perf,
            annot_weights=self.annot_weights[combs["a"]],
            gamma=self.gamma,
            gamma_dist=self.gamma_dist,
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
        x = batch["x"]
        a = batch.get("a", None)
        a = a[0] if a is not None else None
        output = self.forward(x=batch["x"], a=a)
        if a is None:
            return {"p_class": F.softmax(output, dim=-1)}
        else:
            p_class_log = F.log_softmax(output[0], dim=-1)
            p_confusion_log = F.log_softmax(output[1], dim=-1)
            p_confusion_log = p_confusion_log.reshape((len(x), len(a), p_class_log.shape[1], p_class_log.shape[1]))
            p_perf = (p_class_log[:, None, :, None] + p_confusion_log).exp().diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            return {"p_class": p_class_log.exp(), "p_perf": p_perf}

    @torch.no_grad()
    def on_train_epoch_end(self):
        """
        Prints the annotator weights at the end of each epoch, if `self.verbose=True`.
        """
        if self.verbose and self.annot_weights is not None:
            annotator_weights_str = ""
            for w_idx, w in enumerate(self.annot_weights):
                annotator_weights_str += f"|{w:.2f}|"
                if (w_idx + 1) % 10 == 0:
                    annotator_weights_str += "\n"
            annotator_weights_str = annotator_weights_str.replace("||", "|")
            print(f"\n Annotator weights:\n{annotator_weights_str}")
            print(self.annot_weights.sum())
            if self.gamma is not None:
                print(f"\ngamma: {self.gamma}")

    @staticmethod
    def loss(
        z: torch.tensor,
        logits_class: torch.tensor,
        logits_perf: torch.tensor,
        annot_weights: torch.tensor,
        gamma: torch.tensor,
        gamma_dist: Gamma,
    ):
        """
        Computes the loss of MaDL according to the article [1].

        Parameters
        ----------
        z : torch.tensor of shape (n_samples, n_annotators)
            Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
            `i`. A missing label is denoted by `c=-1`.
        logits_class : torch.tensor of shape (n_samples,, n_classes)
            Estimated class-membership logits, where `logits_class[i,c]` refers to the estimated logit for sample `i`,
            and class `c`.
        logits_perf : torch.tensor of shape (n_samples, n_annotators, n_classes)
            Estimated performances logits, where `logits_annot[i,j,c,k]` refers to the estimated logit for sample `i`,
            annotator `j`, true class `c`, and annotation `k`.
        annot_weights : torch.tensor of shape (n_annotators,)
            Annotator weights, where `annot_weights[j]` refers to the weight of annotator `j`.
        gamma : positive float, optional (default=1.25)
            Bandwidth parameter of the radial basis function or cosine kernel.
            If it is None, no regularization is applied
        gamma_dist : positive float, optional (default=1.25)
            Gamma distribution to regularize the value of `gamma` as bandwidth parameter. If it is None,
            no regularization is applied.

        Returns
        -------
        loss : torch.Float
            Computed cross-entropy loss with regularization and annotator weights.

        References
        ----------
        [1] Herde, Marek, Huseljic, Denis, and Sick, Bernhard. "Multi-annotator Deep Learning: A Modular Probabilistic
            Framework for Classification." Trans. Mach. Learn. Res., 2023.
        """
        p_class_log = F.log_softmax(logits_class, dim=-1)
        p_perf_log = F.log_softmax(logits_perf, dim=-1)
        p_annot_log = torch.logsumexp(p_class_log[:, :, None] + p_perf_log, dim=1)

        # Compute prediction loss: cf. left summand of Eq. (25) in the article [1].
        loss = F.nll_loss(p_annot_log, z, reduction="none", ignore_index=-1)
        if annot_weights is not None:
            loss = (loss * annot_weights[None, :]).sum()
        else:
            loss = loss.sum()

        # Normalize loss according to the number of labels: cf. Eq. (26) in article [1].
        n_labels = (z != -1).sum().float().item()
        loss = loss / n_labels

        # Compute loss w.r.t. to the bandwidth parameter `gamma` of the kernel: cf. right summand of Eq. (25) in the
        # article [1].
        if gamma is not None and gamma_dist is not None:
            loss -= gamma_dist.log_prob(torch.clamp(gamma, min=1e-3))

        return loss

    @torch.no_grad()
    def get_gt_parameters(self):
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
    def get_ap_parameters(self):
        """
        Returns the list of parameters of the AP model.

        Returns
        -------
        ap_parameters : list
            The list of the AP models' parameters.
        """
        ap_parameters = list(self.ap_embed_x.parameters())
        ap_parameters += list(self.ap_embed_a.parameters())
        ap_parameters += list(self.ap_hidden.parameters())
        ap_parameters += list(self.ap_outer_product.parameters())
        ap_parameters += list(self.ap_output.parameters())
        return ap_parameters


def xavier_linear(shape):
    """
    Create a tensor with given shape, initial with Xavier uniform weights, and convert to nn.Parameter

    Parameters
    ----------
    shape : tuple
        Shape of the weights to be created.

    Returns
    -------
    weights: nn.Parameter of shape (shape)
        Xavier uniform weights as `nn.Parameter`.
    """
    weights = torch.empty(shape)
    nn.init.xavier_uniform_(weights)
    weights = nn.Parameter(weights)
    return weights


class OuterProduct(nn.Module):
    """OuterProduct

    Outer product of embedded vectors, originally used in the product-based neural network (PNN) model [1]. The
    implementation is based on the GitHub repository  and is associated to the paper [3].

    Parameters
    ----------
    embedding_size : int
        Length of embedding vectors in input; all inputs are assumed to be embedded values.
    output_size : int, optional (default=10)
        Size of output after product and transformation.
    device : string or torch.device, optional (default="cpu")
        Device for storing and computations.

    References
    ----------
    [1] Qu, Y., Cai, H., Ren, K., Zhang, W., Yu, Y., Wen, Y. and Wang, J., 2016, December. Product-based neural
        networks for user response prediction. IEEE Int. Conf. Data Mining (ICDM)
        (pp. 1149-1154). IEEE.
    [2] Fiedler, James (GitHub username: jrfiedler). GitHub Repository: https://github.com/jrfiedler/xynn.
    [3] Fiedler, James. "Simple modifications to improve tabular neural networks." arXiv:2108.03214 (2021).
    """

    def __init__(self, embedding_size, output_size=10, device="cpu"):
        super().__init__()
        self.weights = xavier_linear((output_size, embedding_size, embedding_size))
        self.to(device=device)

    def forward(self, x):
        """Outer product transformation of the input.

        Parameters
        ----------
        x : torch.tensor of shape (batch_size, n_fields, embedding_size)
            Input to be transformed.

        Return
        ------
        op : torch.tensor of shape (batch_size, output_size)
            Input after outer product transformation.
        """
        # r = # batch size
        # f = # fields
        # e, m = embedding size (two letters are needed)
        # p = product output size
        f_sigma = x.sum(dim=1)  # rfe -> re
        p = torch.einsum("re,rm->rem", f_sigma, f_sigma)
        op = torch.einsum("rem,pem->rp", p, self.weights)
        return op
