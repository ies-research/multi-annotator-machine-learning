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
            self.z = torch.full((len(self.x), self.get_n_annotators()), fill_value=-1)
            annotation_file = os.path.join(root, CIFAR10N.annotations_filename)
            annotations = torch.load(annotation_file)
            print(type(annotations))
            z_worse = annotations["worse_label"]
            z_random = np.column_stack((
                annotations["random_label1"],
                annotations["random_label2"],
                annotations["random_label3"])
            )
            worst_indices = np.argmax(z_worse[:, None] == z_random, axis=-1)
            print((z_random[np.arange(len(self.x)), worst_indices] == z_worse).mean())
            worst_annotator_ids = annotator_ids[np.arange(len(self.x)), worst_indices]
            worst_annotator_ids = np.unique(worst_annotator_ids, return_inverse=True)[1]
            print(worst_annotator_ids.max())
            self.z[np.arange(len(self.x)), worst_annotator_ids] = torch.from_numpy(z_worse)
            is_lbld = self.z != -1
            print(is_lbld.float().sum(-1).mean())
            print(((self.z != self.y[:, None]).float() * is_lbld.float()).sum() / is_lbld.float().sum())
            print(is_lbld.float().sum(0).mean())
        elif version in ["valid", "test"]:
            valid_indices, test_indices = train_test_split(
                torch.arange(len(self.x)), train_size=1000, random_state=0, stratify=self.y
            )
            version_indices = valid_indices if version == "valid" else test_indices
            self.x, self.y = self.x[version_indices], self.y[version_indices]

        print(version)
        print(len(self.y))
        print(len(np.unique(self.y)))

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg = self.aggregate_annotations(z=self.z, y=self.y, aggregation_method=aggregation_method)

    def __len__(self):
        return len(self.x)

    def get_n_classes(self):
        return 10

    def get_n_annotators(self):
        return 733

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
