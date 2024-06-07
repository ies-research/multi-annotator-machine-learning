import math
import torch

from typing import Optional, Union


def rbf_kernel(
    A: torch.tensor,
    B: Optional[torch.tensor] = None,
    gamma: Optional[Union[float, torch.tensor]] = None,
):
    """
    Compute radial basis function (RBF) kernel between each row of matrix `A` with each row of matrix `B`.

    Parameters
    ----------
    A : torch.Tensor of shape (n_rows_A, n_cols_A)
        First matrix.
    B : torch.Tensor of shape (n_rows_B, n_cols_B)
        Second matrix. If `B=None`, the similarities are computed between each pair of rows in matrix `A`.
    gamma : float >= 0,
        Bandwidth controlling the width of the RBF kernel. If `None`, we use the mean bandwidth criterion [1] to set a
        default value.

    Returns
    -------
    S : torch.Tensor of shape (n_rows_a * n_rows_b, n_rows_a * n_rows_b)
        Computed similarities via RBF kernel.

    References
    ----------
    [1] Chaudhuri, A., Kakde, D., Sadek, C., Gonzalez, L. and Kong, S., 2017, November. The mean and median criteria
        for kernel bandwidth selection for support vector data description. In 2017 IEEE International Conference on
        Data Mining Workshops (ICDMW) (pp. 842-849). IEEE.
    """
    if gamma is None:
        n_samples = len(A) + (0 if B is None else len(B))
        var_sum = A.var(0).sum() if B is None else torch.vstack((A, B)).var(0).sum()
        s_2 = (2 * n_samples * var_sum) / ((n_samples - 1) * math.log((n_samples - 1) / (2 * 1e-12)))
        gamma = 0.5 / s_2
    B = A if B is None else B
    return torch.exp(-gamma * torch.cdist(A, B) ** 2)


def cosine_kernel(
    A: torch.tensor,
    B: Optional[torch.tensor] = None,
    gamma: Union[float, torch.tensor] = 1.0,
    eps: float = 1e-8,
):
    """
    Compute cosine similarity (normalized to the interval [0, 1]) between each row of matrix `A` with each row of
    matrix `B`.

    Parameters
    ----------
    A : torch.Tensor of shape (n_rows_A, n_cols_A)
        First matrix.
    B : torch.Tensor of shape (n_rows_B, n_cols_B), optional (default=None)
        Second matrix. If `B=None`, the similarities are computed between each pair of rows in matrix `A`.
    gamma : float or int >= 0,
        Bandwidth controlling the width of the cosine similarity based kernel.
    eps : float > 0,
        Positive value to avoid division by zero.

    Returns
    -------
    S : torch.Tensor of shape (n_rows_a * n_rows_b, n_rows_a * n_rows_b)
        Computed cosine similarities.
    """
    B = A if B is None else B
    A_n, B_n = A.norm(dim=1)[:, None], B.norm(dim=1)[:, None]
    A = A / torch.clamp(A_n, min=eps)
    B = B / torch.clamp(B_n, min=eps)
    S = 1 - torch.mm(A, B.transpose(0, 1))
    S = 0.25 * torch.pi * torch.cos(0.5 * torch.pi * (S / gamma))
    return S.clamp(min=0)
