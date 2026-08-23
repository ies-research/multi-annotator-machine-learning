import numpy as np
import torch

from typing import Optional, Union


def dawid_skene_aggregation(
    z: Union[torch.Tensor, np.ndarray],
    n_classes: Optional[int] = None,
    tol: float = 1e-4,
    max_iter: int = 300,
    return_confusion_matrix: bool = False,
):
    """
    Perform Dawid-Skene aggregation on annotator responses using the EM algorithm.

    This function estimates the true labels of items given multiple noisy annotations from different annotators. It
    uses an Expectation-Maximization (EM) approach to update the class marginals and the annotators’ confusion
    matrices.

    Parameters
    ----------
    z : torch.tensor or np.ndarray
        Array of shape (n_samples, n_annotators) containing annotation labels.
        Missing annotations should be represented by -1.
    n_classes : int, optional
        The total number of classes. If not provided, it is inferred from the non-missing
        labels in `z`.
    tol : float, default=1e-4
        Tolerance for convergence of the EM algorithm.
    max_iter : int, default=300
        Maximum number of iterations for the EM algorithm.
    return_confusion_matrix : bool, default=False
        If True, also returns the estimated confusion matrices for each annotator.

    Returns
    -------
    z_agg : torch.tensor
        Tensor of aggregated labels of shape (n_samples,). For items with no annotations,
        the label is -1.
    confusion_matrices : torch.tensor, optional
        If `return_confusion_matrix` is True, also returns an array of shape
        (n_annotators, n_classes, n_classes) containing the confusion matrix for each annotator.

    Notes
    -----
    The algorithm alternates between:
      - **M-step:** Updating the class marginals and the annotators’ confusion matrices.
      - **E-step:** Updating the posterior probabilities for the true class labels.

    The use of vectorized NumPy operations ensures efficiency when handling large datasets.

    References
    ----------
    [1] Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the
        EM algorithm. Journal of the Royal Statistical Society: Series C (Applied Statistics), 28(1), 20-28.
    """
    # Ensure input is a NumPy array.
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
    else:
        z = np.asarray(z)

    n_samples, n_annotators = z.shape

    # Infer the number of classes from non-missing labels if not provided.
    if n_classes is None:
        valid_labels = z[z != -1]
        if valid_labels.size == 0:
            raise ValueError("No valid annotations found to infer number of classes.")
        n_classes = int(valid_labels.max() + 1)

    # If there is no sample with at least two annotations, perform no aggregation.
    if (z != -1).sum(-1).max() <= 1:
        z_agg = torch.from_numpy(z.max(-1)).long()
        if return_confusion_matrix:
            confusion_matrices = torch.stack([torch.eye(n_classes) for _ in range(n_annotators)]).float()
            return z_agg, confusion_matrices
        else:
            return z_agg

    # Initialize a one-hot encoded array for the annotations.
    z_one_hot = np.zeros((n_samples, n_annotators, n_classes), dtype=int)

    # Identify indices with valid (non-missing) annotations.
    valid_mask = z != -1
    rows, cols = np.nonzero(valid_mask)
    z_one_hot[rows, cols, z[rows, cols]] = 1

    # Determine which samples have at least one annotation.
    annotated_mask = z_one_hot.sum(axis=(1, 2)) != 0
    z_one_hot_valid = z_one_hot[annotated_mask]

    # Initialization: Compute initial class probabilities for each annotated sample.
    y_proba = z_one_hot_valid.sum(axis=1, dtype=float)
    y_proba_sum = y_proba.sum(axis=1, keepdims=True)
    y_proba = np.divide(y_proba, y_proba_sum, out=np.zeros_like(y_proba), where=y_proba_sum != 0)

    prev_y_marginals = -np.inf

    confusion_matrices = np.empty((n_annotators, n_classes, n_classes))
    for iteration in range(max_iter):
        # M-step: Update class marginals and confusion matrices.
        y_marginals = np.sum(y_proba, axis=0) / y_proba.shape[0]
        confusion_matrices = np.einsum("nj,nkl->kjl", y_proba, z_one_hot_valid)
        cm_sum = confusion_matrices.sum(axis=-1, keepdims=True)
        confusion_matrices = np.divide(
            confusion_matrices,
            cm_sum,
            out=np.zeros_like(confusion_matrices, dtype=float),
            where=cm_sum != 0,
        )

        # E-step: Update posterior probabilities for each sample.
        # The likelihood is computed by taking the product of the appropriate confusion matrix
        # entries raised to the power of the one-hot annotation (which selects the corresponding entry).
        likelihood = np.prod(confusion_matrices[None, :, :, :] ** z_one_hot_valid[:, :, None, :], axis=(1, 3))
        y_proba = likelihood * y_marginals[None, :]
        y_proba_sum = y_proba.sum(axis=-1, keepdims=True)
        y_proba = np.divide(
            y_proba,
            y_proba_sum,
            out=np.zeros_like(y_proba),
            where=y_proba_sum != 0,
        )

        # Check for convergence.
        if np.abs(y_marginals - prev_y_marginals).sum() < tol:
            break
        prev_y_marginals = y_marginals

    # For samples with annotations, assign the class with the highest posterior probability.
    aggregated_labels = np.full(n_samples, fill_value=-1, dtype=int)
    aggregated_labels[annotated_mask] = np.argmax(y_proba, axis=1)

    # Convert the result back to a PyTorch tensor.
    z_agg = torch.from_numpy(aggregated_labels).long()

    # Replace confusion matrices of annotators without annotations.
    has_no_annotations = (z != -1).sum(0) == 0
    confusion_matrices[has_no_annotations] = np.eye(n_classes)

    if return_confusion_matrix:
        return z_agg, torch.from_numpy(confusion_matrices).float()
    else:
        return z_agg
