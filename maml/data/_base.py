import hashlib
import numpy as np
import torch
import os

from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Literal, Callable, Union
from skactiveml.utils import majority_vote, rand_argmax, compute_vote_vectors

AGGREGATION_METHODS = Optional[Literal["majority-vote", "ground-truth"]]
ANNOTATOR_FEATURES = Optional[Literal["one-hot", "index"]]
TRANSFORMS = Optional[Union[Callable, Literal["auto"]]]
VERSIONS = Literal["train", "valid", "test"]


class MultiAnnotatorDataset(Dataset, ABC):
    """MultiAnnotatorDataset

    Dataset to deal with samples annotated by multiple annotators.
    """

    def __getitem__(self, idx: int):
        batch_dict = {"x": self.get_sample(idx)}
        y = self.get_true_label(idx)
        if y is not None:
            batch_dict["y"] = y
        z = self.get_annotations(idx)
        if z is not None:
            batch_dict["z"] = z
        z_agg = self.get_aggregated_annotation(idx)
        if z_agg is not None:
            batch_dict["z_agg"] = z_agg
        a = self.get_annotators()
        if a is not None:
            batch_dict["a"] = a
        return batch_dict

    @property
    @abstractmethod
    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        pass

    @property
    @abstractmethod
    def get_n_classes(self):
        """
        Returns
        -------
        n_classes : int
            Number of classes.
        """
        pass

    @property
    @abstractmethod
    def get_annotators(self):
        """
        Returns
        -------
        annotators : None or torch.tensor of shape (n_annotators, *)
            Representation of the annotators, e.g., one-hot encoded vectors or metadata.
        """
        pass

    @abstractmethod
    def get_sample(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        sample : torch.tensor
            Sample with the given index.
        """
        pass

    @abstractmethod
    def get_true_label(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        true_label : torch.tensor
            True class label with the given index.
        """
        pass

    @abstractmethod
    def get_annotations(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        annotations : torch.tensor
            Annotations with the given index.
        """
        pass

    @abstractmethod
    def get_aggregated_annotation(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        aggregated_annotation : torch.tensor
            Aggregated annotation with the given index.
        """
        return None

    def __str__(self):
        """
        Provides a summary of dataset statistics.

        Returns
        -------
        stats : str
            Summary of the dataset's statistics.
        """
        stats = "\n############ DATASET SUMMARY ############\n"
        stats += f"n_annotators [#]: {self.get_n_annotators()}\n"
        stats += f"n_samples [#]: {len(self)}\n"
        if hasattr(self, "z") and self.z is not None:
            if self.z.ndim == 3:
                z = self.z.numpy()
                is_not_annotated = np.any(z == -1, axis=-1)
                z = rand_argmax(z, axis=-1, random_state=0)
                z[is_not_annotated] = -1
                z = torch.from_numpy(z)
            else:
                z = self.z
            is_true = (z == self.y[:, None]).float()
            is_lbld = (z != -1).float()
            z_agg = torch.from_numpy(majority_vote(y=z.cpu().numpy(), missing_label=-1, random_state=0))
            n_labels_per_sample = torch.sum(is_lbld, dim=1)
            stats += f"n_labels per sample [#]: {n_labels_per_sample.mean()}+-{n_labels_per_sample.std()}\n"
            n_labels_per_annot = torch.sum(is_lbld, dim=0)
            stats += f"n_labels per annotator [#]: {n_labels_per_annot.mean()}+-{n_labels_per_annot.std()}\n"
            acc = torch.sum(is_true * is_lbld) / torch.sum(is_lbld)
            stats += f"annotation accuracy  [%]: {acc}\n"
            mv_acc = (z_agg == self.y).float().sum() / (z_agg != -1).float().sum()
            stats += f"majority voting accuracy  [%]: {mv_acc}\n"
            acc_per_annot = torch.sum(is_true * is_lbld, dim=0) / n_labels_per_annot
            stats += f"accuracy per annotator [#]: {acc_per_annot.mean()}+-{acc_per_annot.std()}\n"
        return stats

    @staticmethod
    def aggregate_annotations(
        z: torch.tensor, y: Optional[torch.tensor] = None, aggregation_method: Optional[AGGREGATION_METHODS] = None
    ):
        """
        Aggregates the annotations according to a given method.

        Parameters
        ----------
        z : torch.tensor of shape (n_samples, n_annotators) or (n_samples, n_annotators, n_classes)
            Observed annotations, which are class labels in the case of a 2d-array or probabilities in the case of a
            3d-array.
        y : torch.tensor of shape (n_samples,), default=None
            True class labels, which are only required if `aggregation_method="ground-truth"`.
        aggregation_method : str, default=None
            Supported methods are majority voting (`aggregation_method="majority_vote") and returning the true class
            labels (`aggregation_method="ground-truth"). In the case of `aggregation_method=None`, `None` is returned
            as aggregated annotations.

        Returns
        -------
        z_agg : torch.tensor of shape (n_samples,) or None
            Returns the aggregated annotations, if `aggregation_method is not None`.
        """
        if aggregation_method is None:
            return None
        elif aggregation_method == "ground-truth":
            return y
        elif aggregation_method == "majority-vote":
            if z.ndim == 3:
                mask = (z != -1).all(dim=-1, keepdim=True).float()
                clean_prob_tensor = z * mask
                summed_proba = clean_prob_tensor.sum(dim=1)
                proba = summed_proba / summed_proba.sum(dim=-1, keepdim=True)
                class_labels = torch.from_numpy(rand_argmax(proba.numpy(), axis=-1, random_state=0))
            else:
                class_labels = torch.from_numpy(majority_vote(y=z.numpy(), missing_label=-1, random_state=0))
            return class_labels
        elif aggregation_method in ["soft-majority-vote", "selection-frequency"]:
            if z.ndim == 3:
                mask = (z != -1).all(dim=-1, keepdim=True).float()
                clean_prob_tensor = z * mask
                summed_proba = clean_prob_tensor.sum(dim=1)
                proba = summed_proba / summed_proba.sum(dim=-1, keepdim=True)
            else:
                votes = compute_vote_vectors(y=z.numpy(), missing_label=-1)
                proba = torch.from_numpy(votes / votes.sum(axis=-1, keepdims=True))
            if aggregation_method == "soft-majority-vote":
                return proba
            else:
                selection_frequencies, _ = proba.max(dim=-1)
                class_labels = torch.from_numpy(rand_argmax(proba.numpy(), axis=-1, random_state=0))
                is_not_selected = selection_frequencies < 0.7
                class_labels[is_not_selected] = -1
                return class_labels
        else:
            raise ValueError("`aggregation_method` must be in ['majority-vote', 'ground-truth', None].")

    @staticmethod
    def prepare_annotator_features(annotators: ANNOTATOR_FEATURES, n_annotators: int):
        """
        Aggregates the annotations according to a given method.

        Parameters
        ----------
        annotators : None or "index" or "one-hot"
            Defines the representation of the annotators as either indices, one-hot encoded vectors or `None`.
        n_annotators : int
            Number of annotators.

        Returns
        -------
        annotator_features : None or torch.tensor of shape (n_annotators,) or (n_annotators, n_annotators)
            Depending on the parameter `annotators`, the prepared features ar indices, one-hot encoded vectors or
            `None`.
        """
        if annotators == "index":
            return torch.arange(n_annotators)
        elif annotators == "one-hot":
            return torch.eye(n_annotators)
        elif annotators is None:
            return None
        else:
            raise ValueError("`annotators` must be in `['index', 'one-hot', None].")


class SSLDatasetWrapper(MultiAnnotatorDataset):
    """SSLDatasetWrapper

    This class implements an auxiliary dataset wrapper caching self-supervised features of a given self-supervised
    learning (SSL) model.

    Parameters
    ----------
    model : torch.nn.Module
        Self-supervised learning model.
    dataset : MultiAnnotatorDataset
        Multi-annotator dataset whose self-supervised features are to be outputted.
    cache : bool, default = False
        Flag whether the self-supervised features are to be cached.
    cache_dir : str, default=None
        Path to the cache directory for the self-supervised features. Must be a str, if `cache=True`.
    num_hash_samples : int, default=50
        Number of samples used for creating or checking the hash string.
    batch_size : int, default=16
        Batch size to infer the self-supervised features.
    device : "cpu" or "cuda", default="cpu"
        Device to be used for the forward propagation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: MultiAnnotatorDataset,
        cache: bool = False,
        cache_dir: Optional[str] = None,
        num_hash_samples: int = 50,
        batch_size: int = 16,
        device: str = "cpu",
    ):
        self.model = model
        self.dataset = dataset
        self.num_hash_samples = num_hash_samples
        self.batch_size = batch_size
        self.device = device
        if cache:
            if cache_dir is None:
                home_dir = os.path.expanduser("~")
                cache_dir = os.path.join(home_dir, ".cache", "feature_datasets")
            os.makedirs(cache_dir, exist_ok=True)
            hash = self.create_hash_from_dataset_and_model()
            file_name = os.path.join(cache_dir, hash + ".pth")
            if os.path.exists(file_name):
                print("\nLoading cached features from", file_name)
                self.features = torch.load(file_name, map_location="cpu")
            else:
                self.features = self.get_features()
                print("\nSaving features to cache file", file_name)
                torch.save(self.features, file_name)
        else:
            self.features = self.get_features()

    def create_hash_from_dataset_and_model(self):
        """
        Creates and checks the hast string.

        Returns
        -------
        hash : str
            Hash string used for caching.
        """
        hasher = hashlib.md5()

        num_samples = len(self.dataset)
        hasher.update(str(num_samples).encode())

        num_parameters = sum([p.numel() for p in self.model.parameters()])
        hasher.update(str(self.model).encode())
        hasher.update(str(num_parameters).encode())

        indices_to_hash = range(0, num_samples, num_samples // self.num_hash_samples)
        for idx in indices_to_hash:
            sample = self.dataset[idx]["x"]
            hasher.update(str(sample).encode())
        return hasher.hexdigest()

    @torch.no_grad()
    def get_features(self):
        """
        Computes the self-supervised features.

        Returns
        -------
        features : torch.tensor of shape (n_samples,)
            Self-supervised features of the dataset.
        """
        print("\nCache features ...")
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8)
        features = []
        self.model.eval()
        self.model.to(self.device)
        for batch in dataloader:
            features.append(self.model(batch["x"].to(self.device)).to("cpu"))
        features = torch.cat(features)
        return features

    def __len__(self):
        """
        Returns
        -------
        length: int
            Length of the dataset.
        """
        return len(self.dataset)

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return self.dataset.get_n_annotators()

    def get_n_classes(self):
        """
        Returns
        -------
        n_classes : int
            Number of classes.
        """
        return self.dataset.get_n_classes()

    def get_annotators(self):
        """
        Returns
        -------
        annotators : None or torch.tensor of shape (n_annotators, *)
            Representation of the annotators, e.g., one-hot encoded vectors or metadata.
        """
        return self.dataset.get_annotators()

    def get_sample(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        sample : torch.tensor
            Sample with the given index.
        """
        return self.features[idx]

    def get_true_label(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        true_label : torch.tensor
            True class label with the given index.
        """
        return self.dataset.get_true_label(idx)

    def get_annotations(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        true_label : torch.tensor
            True class label with the given index.
        """
        return self.dataset.get_annotations(idx)

    def get_aggregated_annotation(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        aggregated_annotation : torch.tensor
            Aggregated annotation with the given index.
        """
        return self.dataset.get_aggregated_annotation(idx)
