import numpy as np
import torch
import random
from typing import Union, Optional


def permute_same_value_indices(arr: torch.tensor):
    """
    Permutes only the indices of the elements with the same value.

    Parameters
    ----------
    arr : torch.tensor of shape (n_elements,)
        Tensor whose indices are to permuted only for elements with the same value.

    Returns
    -------
    permuted_indices : torch.tensor of shape (n_elements,)
        Randomly permuted indices.
    """
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
    permuted_indices = torch.cat([torch.tensor(index_dict[key]) for key in sorted(index_dict.keys())])

    return permuted_indices


def mixup(
    *arrays,
    alpha: float = 1.0,
    lmbda: Union[None, float, torch.tensor] = None,
    permute_indices: Optional[torch.tensor] = None
):
    """
    Performs mixup [1] for each of the given arrays using the same permutation indices.

    Parameters
    ----------
    arrays : iterable
        A variable number of `torch.tensor` arrays, each with the same length of `n_elements` in the zeroth dimension.
    alpha : float > 0, default=1.0
        Corresponds to the hyperparameter of the beta distribution used for sampling the mixing coefficients. This
        parameter is only used, if `lmbda is None`.
    lmbda : torch.tensor of shape `(n_elements,)`, default=None
        Samples mixing coefficients. If `lmbda is None`, mixing coefficients are sampled using the `alpha` parameter.
    permute_indices : torch.tensor of shape (`n_elements`), default=None
        Permutation indices to be used for mixup. If `permuation_indices is None`, new `permuted_indices` are generated
        by random shuffling.

    Returns
    -------
    outputs : iterable
        Contains the mixed arrays having the same shapes as the original ones.

    References
    ----------
    [1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization.
        Int. Conf. Learn. Represent.
    """
    if lmbda is None:
        lmbda = torch.FloatTensor(np.random.beta(alpha, alpha, size=len(arrays[0]))).to(arrays[0].device)
    if permute_indices is None:
        permute_indices = torch.randperm(len(arrays[0]))
    outputs = []
    for arr in arrays:
        lmbda_arr = lmbda.view(-1, *[1] * (arr.dim() - 1))
        outputs.append(lmbda_arr * arr + (1 - lmbda_arr) * arr[permute_indices])
    outputs.extend([lmbda, permute_indices])
    return tuple(outputs)
