import torch
import numpy as np
import os

from skactiveml.utils import ExtLabelEncoder
from sklearn.model_selection import train_test_split, KFold
from torchvision.datasets.utils import download_and_extract_archive

from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class SentimentPolarity(MultiAnnotatorDataset):
    """SentimentPolarity

    The SentimentPolarity [1] dataset features about 10,500 text files of 2 classes, which have been annotated by 203
    annotators with an accuracy of about 79%.

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
    variant :"worst-1" or "worst-2" or "worst-var" or "rand-1" or "rand-2" or "rand-3" or "rand-2" or "rand-var" or\
            "full"
        Defines subsets of annotations to reflect different learning scenarios.

    References
    ----------
    [1] Rodrigues, F., Pereira, F., & Ribeiro, B. (2013). Learning from Multiple Annotators: Distinguishing Good from
        Random Labelers. Pattern Recognit. Lett., 34(12), 1428-1436.
    """

    base_folder = "sentiment_polarity"
    url = "http://fprodrigues.com//mturk-datasets.tar.gz"
    filename = "MTurkDatasets.tar.gz"

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        download: bool = False,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = None,
        realistic_split: str = "cv-5-0",
        variant: str = "worst-1",
    ):
        # Download data.
        if download:
            download_and_extract_archive(SentimentPolarity.url, root, filename=SentimentPolarity.filename)

        # Check availability of data.
        is_available = os.path.exists(os.path.join(root, SentimentPolarity.base_folder))
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Define label encoder and transforms.
        self.le = ExtLabelEncoder(classes=["pos", "neg"], missing_label="not-available")
        self.transform = transform

        # Load and prepare sample features as numpy arrays.
        folder = os.path.join(root, SentimentPolarity.base_folder)
        data_file = os.path.join(folder, "spc-all-mpnet-base-v2.npz")
        if not os.path.isfile(data_file):
            import pandas as pd
            from sentence_transformers import SentenceTransformer

            df = pd.read_csv(os.path.join(folder, "mturk_answers.csv"), header=0)
            z_df = df.pivot(index="Input.id", columns="WorkerId", values="Answer.sent")
            z_df = z_df.fillna("not-available")
            z = z_df.to_numpy(str)
            df_samples = df.drop_duplicates(subset=["Input.id"]).set_index("Input.id").sort_index()
            x = df_samples.loc[z_df.index, "Input.original_sentence"].to_numpy(str)
            y = df_samples.loc[z_df.index, "Input.true_sent"].to_numpy(str)
            model = SentenceTransformer("all-mpnet-base-v2").encode
            x_list = x.tolist()
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            x = model(x_list, show_progress_bar=True)
            y = self.le.fit_transform(y)
            z = self.le.fit_transform(z)
            np.savez(os.path.join(folder, "spc-all-mpnet-base-v2.npz"), x=x, y=y, z=z)
        data = np.load(os.path.join(folder, "spc-all-mpnet-base-v2.npz"))
        x, y, z = data["x"], data["y"], data["z"]

        # Filter annotations and define number of annotators.
        z = SentimentPolarity.mask_annotations(z=z, y_true=y, variant=variant, n_variants=2, is_not_annotated=z == -1)
        provided_labels = np.sum(z != -1, axis=0) > 0
        z = z[:, provided_labels]
        self.n_annotators = z.shape[-1]

        # Decide for realistic data split or the one with ground truth labels.
        indices = {}
        indices["train"], indices["test"] = train_test_split(
            np.arange(len(y)),
            train_size=3000,
            random_state=0,
            shuffle=True,
            stratify=y,
        )
        if realistic_split is not None:
            if isinstance(realistic_split, float):
                indices["train"], indices["valid"] = train_test_split(
                    indices["train"], train_size=realistic_split, random_state=0
                )
            elif isinstance(realistic_split, str) and realistic_split.startswith("cv"):
                n_splits = int(realistic_split.split("-")[1])
                split_idx = int(realistic_split.split("-")[2])
                k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
                indices["train"], indices["valid"] = list(k_fold.split(indices["train"]))[split_idx]

        # Index according to variant.
        if version not in ["train", "valid", "test"]:
            raise ValueError("`version` must be in `['train', 'valid', 'test']`.")
        self.x = torch.from_numpy(x[indices[version]]).float()
        self.y = torch.from_numpy(y[indices[version]]).long()
        if (version == "train" or realistic_split is not None) and version != "test":
            self.z = torch.from_numpy(z[indices[version]])
        else:
            self.z = None

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg, self.ap_confs = self.aggregate_annotations(
            z=self.z, y=self.y, aggregation_method=aggregation_method
        )

        # Print statistics.
        print(f"variant: {version}")
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
        return 2

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
