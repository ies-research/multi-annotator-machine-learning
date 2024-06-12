import torch
import numpy as np
import os
import pandas as pd
import requests

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class CIFAR10N(MultiAnnotatorDataset):
    """CIFAR10N

    The CIFAR10N [1] dataset features about 60,000 images of 10 classes, which have been annotated by 747 annotators
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
    is_worst : bool, default=False
        Flag whether the variant with the worst annotations is loaded.

    References
    ----------
    [1] Wei, J., Zhu, Z., Cheng, H., Liu, T., Niu, G., & Liu, Y. (2022). Learning with Noisy Labels
        Revisited: A Study Using Real-World Human Annotations. Int. Conf. Learn. Represent.
    """

    url_annotations = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/"
    annotations_filename = "CIFAR-10_human.pt"
    url_side_information = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/"
    side_information_filename = "side_info_cifar10N.csv"

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        download: bool = False,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
        is_worst: bool = False,
    ):
        # Download data.
        cifar10 = CIFAR10(
            root=root,
            train=(version == "train"),
            download=download,
        )
        if download:
            url_list = [
                (CIFAR10N.url_side_information, CIFAR10N.side_information_filename),
                (CIFAR10N.url_annotations, CIFAR10N.annotations_filename),
            ]
            for url, filename in url_list:
                response = requests.get(url=f"{url}/{filename}", params={"downloadformat": "csv"})
                with open(os.path.join(root, filename), mode="wb") as file:
                    file.write(response.content)

        # Set transforms.
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if transform == "auto" and version == "train":
            self.transform = Compose(
                [
                    RandomCrop(32, padding=4),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean, std),
                ]
            )
        elif transform == "auto" and version in ["valid", "test"]:
            self.transform = Compose([ToTensor(), Normalize(mean, std)])
        else:
            self.transform = transform

        # Check availability of data.
        is_available = os.path.exists(os.path.join(root, CIFAR10N.side_information_filename)) and os.path.exists(
            os.path.join(root, CIFAR10N.annotations_filename)
        )
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Set samples and targets.
        self.x = cifar10.data
        self.y = torch.tensor(cifar10.targets).long()

        # Load and prepare annotations as tensor for `version="train"`.
        self.z = None
        if version == "train":
            side_information_file = os.path.join(root, CIFAR10N.side_information_filename)
            annotator_ids = pd.read_csv(side_information_file)[["Worker1-id", "Worker2-id", "Worker3-id"]].values
            annotator_ids = np.repeat(annotator_ids, repeats=10, axis=0).astype(int)
            annotation_file = os.path.join(root, CIFAR10N.annotations_filename)
            annotations = torch.load(annotation_file)
            z_random = np.column_stack(
                (annotations["random_label1"], annotations["random_label2"], annotations["random_label3"])
            )
            if is_worst:
                self.n_annotators = 733
                self.z = torch.full((len(self.x), self.n_annotators), fill_value=-1)
                z_worse = annotations["worse_label"]
                worst_indices = np.argmax(z_worse[:, None] == z_random, axis=-1)
                worst_annotator_ids = annotator_ids[np.arange(len(self.x)), worst_indices]
                worst_annotator_ids = np.unique(worst_annotator_ids, return_inverse=True)[1]
                self.z[np.arange(len(self.x)), worst_annotator_ids] = torch.from_numpy(z_worse)
            else:
                self.n_annotators = 747
                self.z = torch.full((len(self.x), self.n_annotators), fill_value=-1)
                for i in range(z_random.shape[1]):
                    self.z[np.arange(len(self.x)), annotator_ids[:, i]] = torch.from_numpy(z_random[:, i])

        elif version in ["valid", "test"]:
            valid_indices, test_indices = train_test_split(
                torch.arange(len(self.x)), train_size=1000, random_state=0, stratify=self.y
            )
            version_indices = valid_indices if version == "valid" else test_indices
            self.x, self.y = self.x[version_indices], self.y[version_indices]

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg = self.aggregate_annotations(z=self.z, y=self.y, aggregation_method=aggregation_method)

        # Print statistics.
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
        return 10

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return self.n_annotators

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
        x = Image.fromarray(self.x[idx])
        return self.transform(x) if self.transform else x

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
