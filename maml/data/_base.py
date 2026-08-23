import hashlib
import numpy as np
import torch
import os

from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Literal, Callable, Union
from skactiveml.utils import majority_vote, rand_argmax, compute_vote_vectors

from ..utils import dawid_skene_aggregation

AGGREGATION_METHODS = Optional[
    Literal["majority-vote", "ground-truth", "soft-majority-vote", "selection-frequency", "dawid-skene"]
]
ANNOTATOR_FEATURES = Optional[Literal["one-hot", "index"]]
TRANSFORMS = Optional[Union[Callable, Literal["auto"]]]
VERSIONS = Literal["train", "valid", "test"]


class MultiAnnotatorDataset(Dataset, ABC):
    """MultiAnnotatorDataset

    Dataset to deal with samples annotated by multiple annotators.
    """

    def __getitem__(self, idx: int):
        batch_dict = {"idx": idx, "x": self.get_sample(idx)}
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
        if a is not None and z is not None:
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

    def get_annotation_matrix(self):
        """
        Compute the annotation matrix by stacking annotation vectors for all samples.

        This method retrieves annotations for each sample in the dataset using `get_annotations`
        and stacks them into a single 2D tensor. If any call to `get_annotations` returns `None`,
        the method will return `None` immediately.

        Returns
        -------
        z : torch.Tensor or None
            A tensor of shape (n_samples, n_features) containing annotations for all samples,
            or `None` if any sample has no annotations.
        """
        z = []
        for i in range(len(self)):
            z_i = self.get_annotations(i)
            if z_i is None:
                return None
            z.append(z_i)
        return torch.vstack(z)

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
            return None, None
        elif aggregation_method == "ground-truth":
            return y, None
        elif aggregation_method == "majority-vote":
            if z.ndim == 3:
                mask = (z != -1).all(dim=-1, keepdim=True).float()
                clean_prob_tensor = z * mask
                summed_proba = clean_prob_tensor.sum(dim=1)
                proba = summed_proba / summed_proba.sum(dim=-1, keepdim=True)
                class_labels = torch.from_numpy(rand_argmax(proba.numpy(), axis=-1, random_state=0))
            else:
                class_labels = torch.from_numpy(majority_vote(y=z.numpy(), missing_label=-1, random_state=0))
            return class_labels, None
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
                return proba, None
            else:
                selection_frequencies, _ = proba.max(dim=-1)
                class_labels = torch.from_numpy(rand_argmax(proba.numpy(), axis=-1, random_state=0))
                is_not_selected = selection_frequencies < 0.7
                class_labels[is_not_selected] = -1
                return class_labels, None
        elif aggregation_method == "dawid-skene":
            return dawid_skene_aggregation(z=z, return_confusion_matrix=True)
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

    @staticmethod
    def mask_annotations(z, y_true, variant, n_variants, is_not_annotated, class_labels=None):
        """
        Mask annotation entries in a 2D array according to a chosen selection variant.

        This method takes an annotation matrix `z` and applies a masking strategy that
        sets selected entries to -1. It supports multiple variants for selecting
        which annotators to keep or mask:

        - "worst-k": Keep only the k annotators whose labels disagree most with
          the true label `y_true` (using random tie-breaking). All other entries are
          masked.
        - "rand-k": Keep only k randomly chosen annotated entries per sample.
        - "worst-var": Randomly choose a subset of annotated entries of random size
          and keep the ones that most disagree with `y_true`.
        - "rand-var": Randomly choose a subset of annotated entries of random size
          to keep.
        - "full": No masking; returns `z` unchanged.

        Parameters
        ----------
        z : ndarray of shape (n_samples, n_annotators)
            Annotation matrix where each element is a label or annotation value.
        y_true : ndarray of shape (n_samples,)
            Ground-truth labels for each sample.
        variant : {'worst-1', ..., f'worst-{n_variants}',
                   'rand-1', ..., f'rand-{n_variants}',
                   'worst-var', 'rand-var', 'full'}
            Selection variant determining which subset of annotators to keep.
        n_variants : int
            Maximum k for the "worst-k" and "rand-k" variants.
        is_not_annotated : ndarray of bool, shape (n_samples, n_annotators)
            Boolean mask indicating missing annotations (True where missing).
        class_labels : ndarray, shape (n_annotators,), optional
            Array of possible class labels. If None, defaults to unique values of `z`.

        Returns
        -------
        z : ndarray of shape (n_samples, n_annotators)
            The masked annotation matrix with entries set to -1 for masked positions.

        Raises
        ------
        ValueError
            If `variant` is not among the supported options.
        """
        worst_variants = [f"worst-{n+1}" for n in range(n_variants)]
        random_variants = [f"rand-{n+1}" for n in range(n_variants)]
        var_variants = ["rand-var", "worst-var"]
        class_labels = z if class_labels is None else class_labels
        if variant in worst_variants:
            n_annotators_per_sample = int(variant.split("-")[-1])
            is_false = np.full_like(class_labels, fill_value=0.0, dtype=float)
            is_false += (y_true[:, None] != class_labels).astype(float)
            is_false -= 2 * is_not_annotated.astype(float)
            is_not_worst = np.full_like(is_false, fill_value=True, dtype=bool)
            random_floats = np.random.RandomState(n_annotators_per_sample).rand(*is_false.shape)
            worst_indices = np.argsort(-(is_false + random_floats), axis=-1)[:, :n_annotators_per_sample]
            for c in range(n_annotators_per_sample):
                is_not_worst[np.arange(len(is_not_worst)), worst_indices[:, c]] = False
            z[is_not_worst] = -1
        elif variant in random_variants:
            n_annotators_per_sample = int(variant.split("-")[-1])
            is_annotated = (~is_not_annotated).astype(float)
            is_not_selected = np.full_like(is_annotated, fill_value=True, dtype=bool)
            random_floats = np.random.RandomState(n_annotators_per_sample + 4).rand(*is_annotated.shape)
            random_indices = np.argsort(-(is_annotated + random_floats), axis=-1)[:, :n_annotators_per_sample]
            for c in range(n_annotators_per_sample):
                is_not_selected[np.arange(len(is_annotated)), random_indices[:, c]] = False
            z[is_not_selected] = -1
        elif variant in var_variants:
            random_state = np.random.RandomState(0)
            for i in range(len(is_not_annotated)):
                if is_not_annotated[i].all():
                    continue
                # Get the indices of ones in the current row
                annotated_indices = np.where(~is_not_annotated[i])[0]

                # Determine the size of the subset to set to zero
                subset_size = random_state.randint(0, len(annotated_indices))

                # Select worst indices to set to zero
                if variant == "worst-var":
                    is_false = class_labels[i][annotated_indices] == y_true[i]
                    random_floats = random_state.rand(*is_false.shape)
                    worst_indices = np.argsort(-(is_false + random_floats), axis=-1)[:subset_size]
                    indices_to_true = annotated_indices[worst_indices]
                else:
                    # Randomly select indices to set to zero
                    indices_to_true = random_state.choice(annotated_indices, size=subset_size, replace=False)

                # Set the selected indices to zero
                is_not_annotated[i, indices_to_true] = True
            z[is_not_annotated] = -1
        elif variant == "full":
            pass
        else:
            raise ValueError(
                f"`variant` must be in {worst_variants + random_variants + var_variants}, got '{variant}' instead."
            )
        return z


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

    def __getattr__(self, item):
        if "dataset" in self.__dict__:
            return getattr(self.dataset, item)
        else:
            return getattr(self.dataset, item)
