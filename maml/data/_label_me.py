import numpy as np
import os
import pandas as pd
import torch

from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
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

from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class LabelMe(MultiAnnotatorDataset):
    """LabelMe

    The LabelMe [1] dataset features about 2,700 images of 8 classes, which have been annotated by 59 annotators
    with an accuracy of about 74%.

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
    [1] Rodrigues, F., & Pereira, F. Deep Learning from Crowds. AAAI Conf. Artif. Intell.
    """

    base_folder = "LabelMe"
    url = "http://fprodrigues.com/deep_LabelMe.tar.gz"
    filename = "LabelMe.tar.gz"

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
        if download:
            download_and_extract_archive(LabelMe.url, root, filename=LabelMe.filename)

        # Check availability of data.
        is_available = os.path.exists(os.path.join(root, LabelMe.base_folder))
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Load and prepare sample features as tensor.
        folder = os.path.join(root, LabelMe.base_folder)
        if version not in ["train", "valid", "test"]:
            raise ValueError("`version` must be in `['train', 'valid', 'test']`.")
        self.filenames = pd.read_csv(os.path.join(folder, f"filenames_{version}.txt"), header=None).values.ravel()
        for f_idx, filename in enumerate(self.filenames):
            class_directory = filename.split("_")[0]
            self.filenames[f_idx] = os.path.join(folder, version, class_directory, filename)

        # Load and prepare true labels as tensor.
        self.y = pd.read_csv(os.path.join(folder, f"labels_{version}.txt"), header=None).values.ravel()
        self.y = torch.from_numpy(self.y.astype(np.int64))

        # Load and prepare annotations as tensor.
        self.z = None
        if version == "train":
            self.z = pd.read_csv(os.path.join(folder, f"answers.txt"), header=None, sep=" ").values
            provided_labels = np.sum(self.z != -1, axis=0) > 0
            self.z = self.z[:, provided_labels]
            self.z = torch.from_numpy(self.z.astype(np.int64))

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
        return len(self.y)

    def get_n_classes(self):
        """
        Returns
        -------
        n_classes : int
            Number of classes.
        """
        return 8

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return 59

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
        x = Image.open(self.filenames[idx]).convert("RGB")
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
