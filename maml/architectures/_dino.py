import torch
from torch import nn
from typing import Union
from copy import deepcopy
from ._tabnet import gt_tabnet


def gt_dino(
    n_classes: int,
    repo_or_dir: str = "facebookresearch/dinov2",
    model: str = "dinov2_vits14",
    n_hidden_neurons: Union[int, list] = 128,
    dropout_rate: float = 0.0,
    freeze_backbone: bool = True,
):
    """
    Creates a ground truth (GT) model with a DINOv2 ViT [1] architecture.

    Parameters
    ----------
    n_classes : int
        Number of classes used for the classification head implemented as multi-layer perceptron (MLP) or linear layer.
    repo_or_dir : str, default="facebookresearch/dinov2"
        Name of the DINOv2 repository on torch hub.
    model : str, default="dinov2_vits14"
        Name of the DINOv2 architecture on torch hub.
    n_hidden_neurons : int or list, default=129
        Number of hidden neurons of the classification head. In case of an empty list, a linear layer is used as
        classification head.
    dropout_rate : float, default=True
        Dropout rate used, if the classification head is an MLP.
    freeze_backbone : bool, default=True

    Returns
    -------
    gt_embed_x : nn.Module
        Backbone architecture of the DINOv2 model, including potential layers of the MLP.
    gt_output : nn.Module
        Last layer of the created model.
    n_hidden_neurons : int
        Number of neurons in the model's penultimate layer.

    References
    ----------
    [1] Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., ... & Bojanowski, P. (2023).
        DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research.
    """
    dino = torch.hub.load(repo_or_dir=repo_or_dir, model=model)
    gt_tabnet_embed_x, gt_output, n_hidden_neurons = gt_tabnet(
        n_classes=n_classes,
        n_hidden_neurons=n_hidden_neurons,
        n_features=dino.embed_dim,
        dropout_rate=dropout_rate,
    )

    def gt_embed_x():
        if freeze_backbone:
            for param in dino.parameters():
                param.requires_grad = False
        gt_embed_x_module = nn.Sequential(deepcopy(dino), gt_tabnet_embed_x())
        return gt_embed_x_module

    return gt_embed_x, gt_output, n_hidden_neurons
