import numpy as np
import torch
import random
from typing import Union, Optional


def permute_same_value_indices(arr):
    # Create a dictionary to store indices of elements with the same value
    index_dict = {}

    # Populate index_dict
    for i, value in enumerate(arr):
        if value.item() not in index_dict:
            index_dict[value.item()] = []
        index_dict[value.item()].append(i)

    # Shuffle indices for each group of elements with the same value
    for key in index_dict:
        random.shuffle(index_dict[key])

    # Concatenate shuffled indices
    shuffled_indices = torch.cat([torch.tensor(index_dict[key]) for key in sorted(index_dict.keys())])

    return shuffled_indices


def mixup(
        *arrays,
        alpha: float = 1.0,
        lmbda: Union[None, float, torch.tensor] = None,
        permute_indices: Optional[torch.tensor] = None
):
    if lmbda is None:
        lmbda = torch.FloatTensor(np.random.beta(alpha, alpha, size=len(arrays[0]))).to(arrays[0].device)
    if permute_indices is None:
        permute_indices = torch.randperm(len(arrays[0]))
    outputs = []
    for arr in arrays:
        lmbda_arr = lmbda.view(-1, *[1]* (arr.dim() - 1))
        outputs.append(lmbda_arr * arr + (1 - lmbda_arr) * arr[permute_indices])
    outputs.extend([lmbda, permute_indices])
    return tuple(outputs)