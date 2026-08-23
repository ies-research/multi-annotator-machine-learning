import pickle

import torch
import numpy as np
import os
import requests

from sklearn.model_selection import train_test_split, KFold
from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class ImageNet15N(MultiAnnotatorDataset):
    """ImageNet15N

    The ImageNet15N [1] dataset features about 60,000 images of 10 classes, which have been annotated by 747 annotators
    with an accuracy of about 82%.

    Parameters
    ----------
    root : str
        Path to the root directory, where the ata is located.
    version : "train" or "valid" or "test", default="train"
        Defines the version (split) of the dataset.
    download : bool, default=False
        Flag whether the dataset will be downloaded.
    annotators : None or "index" or "one-hot" or "metadata"
        Defines the representation of the annotators as either indices, one-hot encoded vectors, or`None`.
    aggregation_method : str, default=None
        Supported methods are majority voting (`aggregation_method="majority_vote") and returning the true class
        labels (`aggregation_method="ground-truth"). In the case of `aggregation_method=None`, `None` is returned
        as aggregated annotations.
    transform : "auto" or torch.nn.Module, default="auto"
        Transforms for the samples, where "auto" used pre-defined transforms fitting the respective version.

    References
    ----------
    [1] Nguyen, Tri, Shahana Ibrahim, and Xiao Fu. "Noisy Label Learning with Instance-Dependent Outliers:
        Identifiability via Crowd Wisdom." Advances in Neural Information Processing Systems 37 (2024): 97261-97298.
    """

    url_training_data = "https://github.com/ductri/COINNet/raw/refs/heads/main/imagenet15/clip_feature_M=100.pkl"
    filename_training = "image_net_15_n_train.pkl"
    url_test_data = "https://github.com/ductri/COINNet/raw/refs/heads/main/imagenet15/clip_feature_M=100_test.pkl"
    filename_test = "image_net_15_n_test.pkl"

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        download: bool = False,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
        realistic_split: str = "cv-5-0",
    ):
        # Determine whether we load the original training subset.
        is_train = (version == "train" or realistic_split is not None) and version != "test"

        # Download data.
        if download:
            url_list = [
                (ImageNet15N.url_training_data, ImageNet15N.filename_training),
                (ImageNet15N.url_test_data, ImageNet15N.filename_test),
            ]
            for url, filename in url_list:
                response = requests.get(url=url, params={"downloadformat": "pkl"})
                with open(os.path.join(root, filename), mode="wb") as file:
                    file.write(response.content)

        # Set transforms.
        self.transform = None if transform == "auto" else transform

        # Check availability of data.
        is_available = os.path.exists(os.path.join(root, ImageNet15N.filename_training)) and os.path.exists(
            os.path.join(root, ImageNet15N.filename_test)
        )
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        file = ImageNet15N.filename_training if is_train else ImageNet15N.filename_test
        with open(os.path.join(root, file), "rb") as file:
            data = pickle.load(file)

        self.x = torch.from_numpy(data["feature"]).float()
        self.y = torch.from_numpy(data["true_label"]).long()
        self.z = torch.from_numpy(data["noisy_label"]).long() if is_train else None

        # Subindex according to selected data split.
        if realistic_split is not None and is_train:
            version_indices = np.arange(len(self.x))
            if isinstance(realistic_split, float):
                train_indices, valid_indices = train_test_split(
                    version_indices, train_size=realistic_split, random_state=0
                )
            elif isinstance(realistic_split, str) and realistic_split.startswith("cv"):
                n_splits = int(realistic_split.split("-")[1])
                split_idx = int(realistic_split.split("-")[2])
                k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
                train_indices, valid_indices = list(k_fold.split(version_indices))[split_idx]
            else:
                raise ValueError("`realistic_split` must be either `cv` or `str`.")

            version_indices = train_indices if version == "train" else valid_indices
            self.x, self.y, self.z = self.x[version_indices], self.y[version_indices], self.z[version_indices]

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg, self.ap_confs = self.aggregate_annotations(
            z=self.z, y=self.y, aggregation_method=aggregation_method
        )

        # Print statistics.
        print(version)
        print(self)

    def __len__(self):
        """
        Returns
        -------
        length: int
            Length of the dataset.
        """
        return len(self.x)

    def get_n_classes(self):
        """
        Returns
        -------
        n_classes : int
            Number of classes.
        """
        return 15

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return 100

    def get_annotators(self):
        """
        Returns
        -------
        annotators : None or torch.tensor of shape (n_annotators, *)
            Representation of the annotators, e.g., one-hot encoded vectors or metadata.
        """
        return self.a

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
        return self.transform(self.x[idx]) if self.transform else self.x[idx]

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
        return self.z[idx] if self.z is not None else None

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
        return self.z_agg[idx] if self.z_agg is not None else None

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
        return self.y[idx] if self.y is not None else None
