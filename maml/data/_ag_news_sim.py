import torch
import os
import numpy as np

from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from typing import Optional, Literal, Union

from ._base import MultiAnnotatorDataset, ANNOTATOR_FEATURES, AGGREGATION_METHODS, TRANSFORMS, VERSIONS
from ..utils import insert_missing_annotations, multisample_from_probs, compute_and_save_text_embeddings


class AGNewsSim(MultiAnnotatorDataset):
    def __init__(
        self,
        root: str,
        download: bool = False,
        version: VERSIONS = "train",
        annotators: ANNOTATOR_FEATURES = None,
        annotation_file: Optional[str] = None,
        annotation_type: Literal["hard", "sampled", "soft"] = "hard",
        n_annotations_per_sample: Union[float, ArrayLike] = -1,
        alpha: float = 1.0,
        beta: float = 3.0,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
    ):
        # Download data.
        data_dir = os.path.join(root, "ag_news_bert_embeddings")
        dir_exists = len(os.listdir(data_dir)) > 0
        if download and not dir_exists:
            compute_and_save_text_embeddings(
                dataset_name="ag_news",
                model_name='bert-base-uncased',
                output_dir=data_dir,
                target_name="label"
            )
        else:
            if not dir_exists:
                raise ValueError("Set `download=True` to download and compute the BERT embeddings of the dataset.")

        version_appendix = "train" if version in ["train", "valid"] else "test"
        self.x = torch.load(os.path.join(data_dir, f"{version_appendix}_x.pt"))
        self.y = torch.load(os.path.join(data_dir, f"{version_appendix}_y.pt"))
        self.transform = None if transform is "auto" else transform
        if version in ["train", "valid"]:
            train, valid = train_test_split(
                np.arange(len(self.y)), test_size=2000, random_state=0, stratify=self.y
            )
            if version == "train":
                self.x, self.y = self.x[train], self.y[train]
                annotation_indices = torch.arange(0, 118000)
            else:
                self.x, self.y = self.x[valid], self.y[valid]
                annotation_indices = torch.arange(118000, 120000)
        else:
            annotation_indices = torch.arange(120000, 127600)

        # Load and prepare annotations as tensor for `version="train"`.
        self.z = None
        if annotation_file is not None:
            z = torch.load(os.path.join(root, annotation_file))[annotation_indices]
            print(z.shape)
            if annotation_type == "hard":
                self.z = z
            elif annotation_type == "sampled":
                self.z = multisample_from_probs(z)
            elif annotation_type == "soft":
                self.z = z
            else:
                raise ValueError("`annotation_type` must be in `['hard', 'sampled', 'soft']`.")
            if n_annotations_per_sample != -1:
                self.z = insert_missing_annotations(
                    z=self.z,
                    n_annotations_per_sample=n_annotations_per_sample,
                    alpha=alpha,
                    beta=beta,
                    seed=2,
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
        return len(self.x)

    def get_n_classes(self):
        return 4

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
