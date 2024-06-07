import torch
import numpy as np
import os
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from ._base import MultiAnnotatorDataset, AGGREGATION_METHODS, ANNOTATOR_FEATURES, TRANSFORMS, VERSIONS


class CIFAR10H(MultiAnnotatorDataset):

    url = "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-raw.zip"
    filename = "cifar10h-raw.zip"

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        download: bool = False,
        n_annotators: int = 100,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
    ):
        # Download data.
        cifar10 = CIFAR10(
            root=root,
            train=(version != "train"),
            download=download,
        )
        if download:
            download_and_extract_archive(CIFAR10H.url, root, filename=CIFAR10H.filename)

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
        is_available = os.path.exists(os.path.join(root, CIFAR10H.filename))
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Set samples and targets.
        self.x = cifar10.data
        self.y = torch.tensor(cifar10.targets).long()

        # Load and prepare annotations as tensor for `version="train"`.
        self.z = None
        self.n_annotators = n_annotators
        if version == "train":
            annotations_file = os.path.join(root, CIFAR10H.filename)
            df = pd.read_csv(annotations_file, header=0)
            annotator_indices = df.annotator_id.unique()
            n_samples = len(df.cifar10_test_test_idx.unique()) - 1
            self.z = torch.full((n_samples, len(annotator_indices)), -1)
            for a_idx in annotator_indices:
                is_a = (df.annotator_id.values == a_idx) & (df.cifar10_test_test_idx != -99999)
                y_a = np.array(df.chosen_label.values[is_a])
                sample_idx_a = df.cifar10_test_test_idx.values[is_a]
                self.z[sample_idx_a, a_idx] = torch.from_numpy(y_a).long()
            n_correct = (self.z == self.y[:, None]).float().sum(dim=0)
            n_labeled = (self.z != -1).float().sum(dim=0)
            mean_performances = n_correct / n_labeled
            sorted_annotators = mean_performances.argsort()
            self.z = self.z[:, sorted_annotators[: self.n_annotators]]
        elif version in ["valid", "test"]:
            valid_indices, test_indices = train_test_split(
                torch.arange(len(self.x)), train_size=500, random_state=0, stratify=self.y
            )
            version_indices = valid_indices if version == "valid" else test_indices
            self.x, self.y = self.x[version_indices], self.y[version_indices]

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg = self.aggregate_annotations(z=self.z, y=self.y, aggregation_method=aggregation_method)

    def __len__(self):
        return len(self.x)

    def get_n_classes(self):
        return 10

    def get_n_annotators(self):
        return self.n_annotators

    def get_annotators(self):
        return self.a

    def get_sample(self, idx):
        x = Image.fromarray(self.x[idx])
        return self.transform(x) if self.transform else x

    def get_annotations(self, idx):
        return self.z[idx] if self.z is not None else None

    def get_aggregated_annotation(self, idx):
        return self.z_agg[idx] if self.z_agg is not None else None

    def get_true_label(self, idx):
        return self.y[idx] if self.y is not None else None
