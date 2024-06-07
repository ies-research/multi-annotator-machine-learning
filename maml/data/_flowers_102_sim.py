import torch
import os

from numpy.typing import ArrayLike
from PIL import Image
from torchvision.datasets import Flowers102
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Resize,
    CenterCrop,
    RandomErasing,
)
from typing import Optional, Literal, Union

from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS
from ..utils import insert_missing_annotations


class Flowers102Sim(MultiAnnotatorDataset):
    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        download: bool = False,
        annotation_file: Optional[str] = None,
        n_annotations_per_sample: Union[float, ArrayLike] = -1,
        alpha: float = 1.0,
        beta: float = 3.0,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
    ):
        # Download data.
        flower102 = Flowers102(
            root=root,
            split="val" if version == "valid" else version,
            download=download,
        )

        # Set transforms.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if transform == "auto" and version == "train":
            self.transform = Compose(
                [
                    Resize(232),
                    RandomResizedCrop(224),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    RandomErasing(),
                    Normalize(mean, std),
                ]
            )
        elif transform == "auto" and version in ["valid", "test"]:
            self.transform = Compose([Resize(232), CenterCrop(224), ToTensor(), Normalize(mean, std)])
        else:
            self.transform = transform

        # Set samples and targets.
        self.filenames = flower102._image_files
        self.y = torch.tensor(flower102._labels).long()
        if version == "train":
            annotation_indices = torch.arange(0, 1020)
        elif version == "valid":
            annotation_indices = torch.arange(1020, 2040)
        else:
            annotation_indices = torch.arange(2040, 8189)

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
                    seed=3,
                )
                print(version)
                is_lbld = self.z != -1
                print(is_lbld.float().sum(-1).mean())
                print(((self.z != self.y[:, None]).float() * is_lbld.float()).sum() / is_lbld.float().sum())

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg = self.aggregate_annotations(z=self.z, y=self.y, aggregation_method=aggregation_method)

    def __len__(self):
        return len(self.filenames)

    def get_n_classes(self):
        return 102

    def get_n_annotators(self):
        return self.z.shape[1] if self.z is not None else 0

    def get_annotators(self):
        return self.a

    def get_sample(self, idx):
        x = Image.open(self.filenames[idx]).convert("RGB")
        return self.transform(x) if self.transform else x

    def get_annotations(self, idx):
        return self.z[idx] if self.z is not None else None

    def get_aggregated_annotation(self, idx):
        return self.z_agg[idx] if self.z_agg is not None else None

    def get_true_label(self, idx):
        return self.y[idx] if self.y is not None else None
