import torch
import os
import numpy as np

from numpy.typing import ArrayLike
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Union

from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS
from ..utils import insert_missing_annotations


class LetterSim(MultiAnnotatorDataset):
    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        annotation_file: Optional[str] = None,
        n_annotations_per_sample: Union[float, ArrayLike] = -1,
        alpha: float = 1.0,
        beta: float = 3.0,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
    ):
        # Download data.
        self.x, self.y = fetch_openml(data_id=6, data_home=root, cache=True, return_X_y=True)
        self.x = self.x.values
        self.y = LabelEncoder().fit_transform(self.y.values)

        # Train and test split data.
        split_dict = {}
        train, split_dict["test"] = train_test_split(
            np.arange(len(self.y)), test_size=0.2, random_state=0, stratify=self.y
        )
        split_dict["train"], split_dict["valid"] = train_test_split(
            train, test_size=500, random_state=0, stratify=self.y[train]
        )

        # Set transforms.
        if transform == "auto":
            sc = StandardScaler().fit(self.x[split_dict["train"]])
            self.x = sc.transform(self.x)
            self.transform = None
        else:
            self.transform = transform

        # Set samples and targets.
        n_samples = len(self.x)
        self.x = self.x[split_dict[version]]
        self.y = self.y[split_dict[version]]
        self.x, self.y = torch.from_numpy(self.x).float(), torch.from_numpy(self.y).long()
        if version == "train":
            annotation_indices = torch.arange(0, len(split_dict["train"]))
        elif version == "valid":
            annotation_indices = torch.arange(len(split_dict["train"]), len(train))
        else:
            annotation_indices = torch.arange(len(train), n_samples)

        # Load and prepare annotations as tensor for `version="train"`.
        self.z = None
        if annotation_file is not None:
            self.z = torch.load(os.path.join(root, annotation_file))[annotation_indices]
            if n_annotations_per_sample != -1:
                self.z = insert_missing_annotations(
                    z=self.z,
                    n_annotations_per_sample=n_annotations_per_sample,
                    alpha=alpha,
                    beta=beta,
                    seed=1,
                )
                is_lbld = self.z != -1
                print(is_lbld.float().sum(-1).mean())
                print(((self.z != self.y[:, None]).float() * is_lbld.float()).sum() / is_lbld.float().sum())

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg = self.aggregate_annotations(z=self.z, y=self.y, aggregation_method=aggregation_method)

    def __len__(self):
        return len(self.x)

    def get_n_classes(self):
        return 26

    def get_n_annotators(self):
        return self.z.shape[1] if self.z is not None else 0

    def get_annotators(self):
        return self.a

    def get_sample(self, idx):
        return self.x[idx] if self.transform is None else self.transform(self.x[idx])

    def get_annotations(self, idx):
        return self.z[idx] if self.z is not None else None

    def get_aggregated_annotation(self, idx):
        return self.z_agg[idx] if self.z_agg is not None else None

    def get_true_label(self, idx):
        return self.y[idx] if self.y is not None else None
