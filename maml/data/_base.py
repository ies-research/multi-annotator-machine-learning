import torch

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Optional, Literal, Callable, Union
from skactiveml.utils import majority_vote

AGGREGATION_METHODS = Optional[Literal["majority-vote", "ground-truth"]]
ANNOTATOR_FEATURES = Optional[Literal["one-hot", "index"]]
TRANSFORMS = Optional[Union[Callable, Literal["auto"]]]
VERSIONS = Literal["train", "valid", "test"]


class MultiAnnotatorDataset(Dataset, ABC):
    """MultiAnnotatorDataset

    Dataset to deal with samples annotated by multiple annotators.
    """

    def __getitem__(self, idx: int):
        batch_dict = {"x": self.get_sample(idx)}
        y = self.get_true_label(idx)
        if y is not None:
            batch_dict["y"] = y
        z = self.get_annotations(idx)
        if z is not None:
            batch_dict["z"] = z
        z_agg = self.get_aggregated_annotation(idx)
        if z_agg is not None:
            batch_dict["z_agg"] = z_agg
        a = self.get_annotators()
        if a is not None:
            batch_dict["a"] = a
        return batch_dict

    @property
    @abstractmethod
    def get_n_annotators(self):
        pass

    @property
    @abstractmethod
    def get_n_classes(self):
        pass

    @property
    @abstractmethod
    def get_annotators(self):
        pass

    @abstractmethod
    def get_sample(self, idx: int):
        pass

    @abstractmethod
    def get_true_label(self, idx: int):
        pass

    @abstractmethod
    def get_annotations(self, idx: int):
        pass

    @abstractmethod
    def get_aggregated_annotation(self, idx: int):
        return None

    @staticmethod
    def aggregate_annotations(
        z: torch.tensor, y: Optional[torch.tensor] = None, aggregation_method: Optional[AGGREGATION_METHODS] = None
    ):
        if aggregation_method is None:
            return None
        elif aggregation_method == "ground-truth":
            return y
        elif aggregation_method == "majority-vote":
            return majority_vote(y=z, missing_label=-1)
        else:
            raise ValueError("`aggregation_method` must be in ['majority-vote', 'ground-truth', None].")

    @staticmethod
    def prepare_annotator_features(annotators: Optional[ANNOTATOR_FEATURES], n_annotators: int):
        if annotators == "index":
            return torch.arange(n_annotators)
        elif annotators == "one-hot":
            return torch.eye(n_annotators)
        elif annotators is None:
            return None
        else:
            raise ValueError("`annotators` must be in `['index', 'one-hot', None].")
