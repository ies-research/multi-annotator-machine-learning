import numpy as np
import torch

from copy import deepcopy
from lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader, Subset
from numpy.typing import ArrayLike
from typing import Optional, Union
from collections.abc import Iterable


def multisample_from_probs(probs: torch.tensor):
    """
    Weighted sampling for given probabilities.

    Parameters
    ----------
    probs : torch.tensor of shape (n_samples, n_classes) or (n_samples, n_annotators, n_classes)
        Probabilities as parameters of categorical distributions, i.e., `probs[i]` defines the parameters of the i-th
        categorical distribution.

    Returns
    -------
    selection: torch.tensor of shape (n_samples,)
        `selection[i]` has been randomly drawn according to a categorical distribution with probabilities `probs[i]`.
    """
    selection = []
    if probs.ndim == 3:
        for i in range(probs.shape[1]):
            rand_values = torch.rand((len(probs[:, i]), 1), device=probs.device)
            cum_probs = probs[:, i].cumsum(dim=1)
            selection.append(torch.searchsorted(cum_probs, rand_values).squeeze(1))
        return torch.stack(selection, dim=1)
    else:
        rand_values = torch.rand((len(probs), 1), device=probs.device)
        cum_probs = probs.cumsum(dim=1)
        return torch.searchsorted(cum_probs, rand_values).squeeze(1)


def insert_missing_annotations(
    z: torch.tensor,
    n_annotations_per_sample: Union[int, ArrayLike] = 1,
    alpha: float = 1.0,
    beta: float = 3.0,
    seed: Optional[int] = None,
):
    """
    Inserts missing annotations to simulate real-world crowd-working datasets.

    Parameters
    ----------
    z : torch.tensor of shape (n_samples, n_annotators)
        Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample `i`.
    n_annotations_per_sample: int or array-like of shape (n_possible_values,), optional (default=1)
        Defines the possible number of annotations. If an array-like object is given, the number of annotations is
        randomly selected from this object for each sample.
    alpha : float, optional (default=1)
        First parameter of the beta distribution to sample the probabilities of selecting an annotator for keeping
        their annotation.
    beta : float, optional (default=3)
        Second parameter of the beta distribution to sample the probabilities of selecting an annotator for keeping
        their annotation.
    seed : int, optional (default=None)
        Seed for reproducibility.

    Returns
    -------
    z : torch.tensor of shape (n_samples, n_annotators)
        Annotations, where `z[i,j]=c` indicates that annotator `j` provided class label `c` for sample
        `i`. A missing label is denoted by `c=-1`.
    """
    n_samples = z.shape[0]
    n_annotators = z.shape[1]
    random_state = np.random.RandomState(seed)
    p_select = random_state.beta(alpha, beta, size=n_annotators)
    p_select /= p_select.sum()
    if isinstance(n_annotations_per_sample, int):
        n_annotations_per_sample = [n_annotations_per_sample]
    n_annotations_per_sample = random_state.choice(n_annotations_per_sample, size=n_samples)
    mask = torch.full_like(z, fill_value=True)
    annot_indices = np.arange(n_annotators)
    for i in range(n_samples):
        selected_annot_indices = random_state.choice(
            annot_indices, p=p_select, size=n_annotations_per_sample[i], replace=False
        )
        mask[i][selected_annot_indices] = False
    z[mask == True] = -1
    return z


def simulate_annotator_classifiers(
    dataset: Dataset,
    model: LightningModule,
    trainer_dict: dict,
    data_loader_dict: dict,
    train_ratios: ArrayLike,
    max_epochs: ArrayLike,
    y: Optional[ArrayLike] = None,
):
    """
    Annotators can be seen as human classifiers. Hence, we use classifiers based on machine learning techniques to
    represent these annotators. Given a data set comprising samples with their true labels, a classifier is trained on
    a subset of sample-label-pairs. Subsequently, this trained classifier is used as proxy of an annotator. As a
    result, the labels for a sample are provided by this classifier.

    Parameters
    ----------
    dataset : Dataset
        Pytorch dataset to be used for classification.
    model : LightningModule
        Model to be fitted on a subset for making predictions.
    trainer_dict : dict
        Dictionary of parameters passed to the `Trainer` being responsible for fitting and predicting.
    data_loader_dict : dict
        Dictionary of parameters passed to the `DataLoader` for training and predicting.
    train_ratios : array-like of shape (n_annotators,) or (n_annotators, n_classes)
        If `train_ratios` is of shape `(n_annotators,)`, the entry `train_ratios[i]` indicates the ratio of samples
        used training the classifier of annotator `j`, e.g., `train_ratios[2]=0.3` means that 30% of the samples are
        used to train the classifier of annotator `2`.
        If `train_ratios` is of shape `(n_annotators, n_classes)`, the entry `train_ratios[i, y]` indicates the ratio
        of samples  used training the classifier of annotator `j`, e.g., `train_ratios[2, 3]=0.3` means that 30% of the
        samples of class `3` are used to train the classifier of annotator `2`.
    max_epochs : array-like of shape (n_annotators,)
        The entry `max_epochs[i]` indicates the maximum number of epochs used training the classifier of annotator `j`,
        e.g., `max_epochs[2]=3` means that classifier of annotator `2` is trained for maximum 3 epochs.
    y : array-like of shape (n_samples)
        If `train_ratios` is of shape `(n_annotators, n_classes)`, the array `y` consisting of the true class labels
        of the dataset's samples must be provided.

    Returns
    -------
    z : torch.tensor of shape (n_samples, n_annotators)
        Class labels provided by simulated annotators.
    """
    data_loader_dict_predict = data_loader_dict.copy()
    data_loader_dict_predict["shuffle"] = False
    dataloader = DataLoader(dataset=dataset, **data_loader_dict_predict)
    data_loader_dict_train = data_loader_dict.copy()
    data_loader_dict_train["drop_last"] = True
    z = []
    for idx, (train_ratio, max_epoch) in enumerate(zip(train_ratios, max_epochs)):
        if isinstance(train_ratio, Iterable):
            subset_indices = []
            for c, tr in enumerate(train_ratio):
                c_idx = torch.argwhere(y == c)
                c_idx = c_idx[torch.randint(low=0, high=len(c_idx), size=(int(tr * len(c_idx)),))]
                subset_indices.append(c_idx)
            subset_indices = torch.concat(subset_indices).ravel()
        else:
            subset_indices = torch.randint(low=0, high=len(dataset), size=(int(train_ratio * len(dataset)),))
        subset = Subset(dataset=dataset, indices=subset_indices)
        dataloader_subset = DataLoader(dataset=subset, **data_loader_dict_train)
        trainer_dict_subset = deepcopy(trainer_dict)
        trainer_dict_subset["max_epochs"] = int(max_epoch)
        trainer_subset = Trainer(**trainer_dict_subset)
        model_subset = deepcopy(model)
        print(f"train_ratio={train_ratio}, max_epochs={max_epoch}")
        trainer_subset.fit(model=model_subset, train_dataloaders=dataloader_subset)
        trainer_subset.test(model=model_subset, dataloaders=dataloader)
        y_pred_list = trainer_subset.predict(model=model_subset, dataloaders=dataloader)
        z.append(torch.concat([y_pred_dict["p_class"] for y_pred_dict in y_pred_list]))
    return torch.stack(z, dim=1)
