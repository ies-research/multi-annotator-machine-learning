import torch
from torch import nn
from ._tabnet import gt_tabnet
from typing import Union
from copy import deepcopy


def gt_dino(
    n_classes: int,
    repo_or_dir: str = "facebookresearch/dinov2",
    model: str = "dinov2_vits14",
    n_hidden_neurons: Union[int, list] = 128,
    dropout_rate: float = 0.0,
    freeze_backbone: bool = True,
):
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
