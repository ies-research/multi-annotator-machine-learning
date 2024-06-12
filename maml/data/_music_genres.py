import torch
import numpy as np
import os
import pandas as pd

from skactiveml.utils import ExtLabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive


from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class MusicGenres(MultiAnnotatorDataset):
    """MusicGenres

    The MusicGenres [1] dataset features 1,000 sound files of 10 classes, which have been annotated by 41 annotators
    with an accuracy of about 56%.

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
    [1] Rodrigues, F., Pereira, F., & Ribeiro, B. (2013). Learning from Multiple Annotators: Distinguishing Good from
        Random Labelers. 	Pattern Recognit. Lett., 34(12), 1428-1436.
    """

    base_folder = "music_genre_classification"
    url = "http://fprodrigues.com//mturk-datasets.tar.gz"
    filename = "MTurkDatasets.tar.gz"

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
            download_and_extract_archive(MusicGenres.url, root, filename=MusicGenres.filename)

        # Check availability of data.
        is_available = os.path.exists(os.path.join(root, MusicGenres.base_folder))
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Load and prepare sample features as tensors.
        folder = os.path.join(root, MusicGenres.base_folder)
        sc = None
        if transform == "auto":
            df = pd.read_csv(os.path.join(folder, "music_genre_gold.csv"), header=0)
            sc = StandardScaler().fit(df.values[:, 1:-1].astype(np.float32))
        if version == "train":
            df = pd.read_csv(os.path.join(folder, "music_genre_gold.csv"), header=0)
        elif version in ["valid", "test"]:
            df = pd.read_csv(os.path.join(folder, "music_genre_test.csv"), header=0)
            valid_indices, test_indices = train_test_split(
                np.arange(len(df)), train_size=100, random_state=0, stratify=df["class"].values
            )
            df = df.iloc[valid_indices] if version == "valid" else df.iloc[test_indices]
        else:
            raise ValueError("`version` must be in `['train', 'valid', 'test']`.")
        self.x = df.values[:, 1:-1].astype(np.float32)

        # Set transforms.
        if isinstance(sc, StandardScaler):
            self.x = sc.transform(self.x)
            self.transform = None
        else:
            self.transform = transform
        self.x = torch.from_numpy(self.x)

        # Setup label encoder.
        self.le = ExtLabelEncoder(
            classes=[
                "blues",
                "classical",
                "country",
                "disco",
                "hiphop",
                "jazz",
                "metal",
                "pop",
                "reggae",
                "rock",
            ],
            missing_label="not-available",
        )

        # Load and prepare annotations as tensor.
        self.z = None
        n_annotators = 44
        if version == "train":
            df_answers = pd.read_csv(os.path.join(folder, "music_genre_mturk.csv"), header=0)
            annotator_indices = df_answers["annotator"].unique()
            self.z = np.full((len(df), n_annotators), fill_value="not-available").astype(str)
            for row_idx, row in df_answers.iterrows():
                sample_idx = np.where(df["id"].values == row["id"])[0][0]
                annotator_idx = np.where(annotator_indices == row["annotator"])[0][0]
                self.z[sample_idx, annotator_idx] = row["class"]
            self.z = torch.from_numpy(self.le.fit_transform(self.z).astype(np.int64))

        # Load and prepare true labels as tensor.
        self.y = df["class"].values.astype(str)
        self.y = torch.from_numpy(self.le.fit_transform(self.y).astype(np.int64))

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
        return 44

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
