import torch

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Literal, Optional
from ._resnet import gt_resnet
from ._tabnet import gt_tabnet
from ._dino import gt_dino
from ..classifiers import AnnotMixModule, OuterProduct

CLASSIFIER_NAMES = Literal[
    "aggregate",
    "annot_mix",
    "conal",
    "crowdar",
    "madl",
    "trace_reg",
    "geo_reg_f",
    "geo_reg_w",
    "union_net",
    "crowd_layer",
]
ARCHITECTURE_NAMES = Literal["resnet", "tabnet", "dino"]


def gt_net(gt_name: ARCHITECTURE_NAMES, gt_params_dict: dict):
    """
    Creates the architecture of the ground truth (GT) model.

    Parameters
    ----------
    gt_name : str
        Name of the GT model's architecture (cf. ARCHITECTURE_NAMES).
    gt_params_dict : dict
        Dictionary of parameters used for creating the respective GT models' architecture.

    Returns
    -------
    gt_model : nn.Module
        GT model's architecture as a Pytorch module.
    """
    # Get building block of the GT model.
    if gt_name == "resnet":
        gt_net_func = gt_resnet
    elif gt_name == "tabnet":
        gt_net_func = gt_tabnet
    elif gt_name == "dino":
        gt_net_func = gt_dino
    else:
        raise ValueError()
    return gt_net_func(**gt_params_dict)


def maml_net_params(
    gt_name: ARCHITECTURE_NAMES,
    gt_params_dict: dict,
    classifier_name: CLASSIFIER_NAMES,
    optimizer: Optimizer.__class__,
    n_annotators: int = 0,
    annotators: Optional[torch.tensor] = None,
    classifier_specific: Optional[dict] = None,
    optimizer_gt_dict: Optional[dict] = None,
    optimizer_ap_dict: Optional[dict] = None,
    lr_scheduler: Optional[LRScheduler.__class__] = None,
    lr_scheduler_dict: Optional[dict] = None,
    embed_size: Optional[int] = None,
):
    """
    Defines the parameters passed to the class of the corresponding MAML classifier.

    Parameters
    ----------
    gt_name : str
        Name of the architecture of the GT model.
    gt_params_dict : dict
        Parameters passed to the function to setup the architecture of the GT model.
    classifier_name : str
        Name of the MAML classifier whose parameters are to be defined.
    optimizer : Optimizer.__class__
        Defines the class of the optimizer to be used.
    n_annotators : int, optional (default=0)
        Defines the number of annotators.
    annotators : torch.tensor, optional (default=None)
        Defines a tensor to represent annotators. The representation depends on the specific MAML classifier.
    classifier_specific : dict, optional (default=None)
        Defines a dictionary of parameters, which a specific for the corresponding MAML classifier.
    optimizer_gt_dict: : dict, optional (default=None)
        Defines the dictionary of parameters for optimizing the GT model.
    optimizer_ap_dict : dict, optional (default=None)
        Defines the dictionary of parameters for optimizing the AP model.
    lr_scheduler : LRScheduler.__class__, optional (default=None)
        Defines the learning rate scheduler controlling the learning of the GT and AP model.
    lr_scheduler_dict : dict, optional (default=None)
        Defines the dictionary of parameters of the learning rate scheduler.
    embed_size : int, optional (default=None)
        Defines the embedding size (dimensionality) to create the architecture of AP models of certain MAML
        classifiers.

    Returns
    -------
    params_dict : dict
        Dictionary of parameters to be passed to the class of the MAML classifier with the name `classifier_name`.
    """
    # Get building block of the GT model.
    gt_embed_x, gt_output, n_hidden_neurons = gt_net(gt_name, gt_params_dict)

    # Define parameters of the AP model.
    n_classes = gt_params_dict["n_classes"]
    params_dict = _ap_net_params(
        n_classes=n_classes,
        n_annotators=n_annotators,
        annotators=annotators,
        classifier_name=classifier_name,
        n_hidden_neurons=n_hidden_neurons,
        embed_size=embed_size,
    )

    classifier_specific = dict(classifier_specific) if classifier_specific is not None else {}

    # Define parameters of the GT model.
    if classifier_name == "annot_mix":
        params_dict["network"] = AnnotMixModule(
            n_classes=n_classes,
            gt_embed_x=gt_embed_x(),
            gt_output=gt_output(),
            ap_embed_a=params_dict.pop("ap_embed_a")(),
            ap_embed_x=params_dict.pop("ap_embed_x")(),
            ap_hidden=params_dict.pop("ap_hidden")(),
            ap_output=params_dict.pop("ap_output")(),
        )
    else:
        params_dict["gt_embed_x"] = gt_embed_x()
        params_dict["gt_output"] = gt_output()

    # Add parameters specific for the respective multi-annotator classifier.
    if classifier_name in ["aggregate", "madl"]:
        params_dict["n_classes"] = n_classes
    elif classifier_name == "conal":
        params_dict["n_classes"] = n_classes
        params_dict["annotators"] = annotators
    elif classifier_name in ["trace_reg", "geo_reg_f", "geo_reg_w", "crowdar", "union_net", "crowd_layer"]:
        params_dict["n_classes"] = n_classes
        params_dict["n_annotators"] = n_annotators
    params_dict["optimizer"] = optimizer
    params_dict["optimizer_gt_dict"] = optimizer_gt_dict
    if classifier_name != "aggregate":
        params_dict["optimizer_ap_dict"] = optimizer_ap_dict
    params_dict["lr_scheduler"] = lr_scheduler
    params_dict["lr_scheduler_dict"] = lr_scheduler_dict
    params_dict.update(classifier_specific)
    return params_dict


def _ap_net_params(
    classifier_name: CLASSIFIER_NAMES,
    n_classes: int,
    n_annotators: int,
    n_hidden_neurons: int,
    annotators: Optional[torch.tensor] = None,
    embed_size: Optional[int] = None,
):
    params_dict = {}
    if n_annotators == 0:
        return params_dict

    def ap_embed_a():
        annot_dim = annotators.shape[1] if annotators is not None else n_annotators
        return nn.Linear(in_features=annot_dim, out_features=embed_size)

    def ap_embed_x():
        return nn.Linear(in_features=n_hidden_neurons, out_features=embed_size)

    def ap_hidden():
        if classifier_name == "madl":
            return nn.Sequential(
                nn.Linear(in_features=3 * embed_size, out_features=128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=embed_size),
                nn.BatchNorm1d(embed_size),
            )
        else:
            return nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=2 * embed_size, out_features=embed_size),
                nn.BatchNorm1d(embed_size),
                nn.ReLU(),
            )

    def ap_output():
        return nn.Linear(in_features=embed_size, out_features=n_classes**2)

    def ap_outer_product():
        return OuterProduct(embedding_size=embed_size, output_size=embed_size)

    if classifier_name in ["conal", "madl"]:
        params_dict["ap_embed_a"] = None if classifier_name == "aggregate" else ap_embed_a()
        params_dict["ap_embed_x"] = None if classifier_name == "aggregate" else ap_embed_x()
        if classifier_name in ["madl"]:
            params_dict["ap_hidden"] = None if classifier_name == "aggregate" else ap_hidden()
            params_dict["ap_output"] = None if classifier_name == "aggregate" else ap_output()
            if classifier_name == "madl":
                params_dict["ap_outer_product"] = ap_outer_product()
    elif classifier_name == "crowdar":
        params_dict["gt_embed_dim_x"] = n_hidden_neurons
    elif classifier_name == "annot_mix":
        params_dict["ap_embed_a"] = ap_embed_a
        params_dict["ap_embed_x"] = ap_embed_x
        params_dict["ap_hidden"] = ap_hidden
        params_dict["ap_output"] = ap_output
    return params_dict
