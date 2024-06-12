from torch import nn
from typing import Literal
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def gt_resnet(n_classes: int, variant: Literal[18, 34, 50, 101, 152], pretrained: bool = False, is_32_32: bool = True):
    """
    Creates a ground truth (GT) model with a ResNet [1] architecture.

    Parameters
    ----------
    n_classes : int
        Number of classes used for the classification head implemented as multi-layer perceptron (MLP) or linear layer.
    variant : int
        ResNet variant to be created.
    pretrained : bool, default=False
        Flag whether a pretrained ResNet is to be used.
    is_32_32 : bool, default=False
        Flag whether the architecture is to fit 32x32, e.g., to be used for datasets, such as CIFAR10.

    Returns
    -------
    gt_embed_x : nn.Module
        Backbone architecture of the ResNet model.
    gt_output : nn.Module
        Last layer of the created model.
    n_hidden_neurons : int
        Number of neurons in the model's penultimate layer.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. IEEE Conf. Comput.
        Vis. Pattern Recognit. (pp. 770-778).
    """
    n_hidden_neurons = 512 if variant in [18, 34] else 2048

    def gt_embed_x():
        if variant == 18:
            resnet = resnet18(pretrained=pretrained)
        elif variant == 34:
            resnet = resnet34(pretrained=pretrained)
        elif variant == 50:
            resnet = resnet50(pretrained=pretrained)
        elif variant == 101:
            resnet = resnet101(pretrained=pretrained)
        elif variant == 152:
            resnet = resnet152(pretrained=pretrained)
        else:
            raise ValueError(
                f"`variant = {variant}` is invalid since only the integers `[18, 34, 52, 152]` are allowed."
            )
        if is_32_32:
            resnet.conv1 = nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
        resnet.maxpool = nn.Identity()
        children_list = []
        for n, c in resnet.named_children():
            children_list.append(c)
            if n == "avgpool":
                break
        children_list.append(nn.Flatten())
        return nn.Sequential(*children_list)

    def gt_output():
        return nn.Linear(n_hidden_neurons, n_classes)

    return gt_embed_x, gt_output, n_hidden_neurons
