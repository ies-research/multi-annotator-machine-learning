from ._base import MultiAnnotatorDataset, SSLDatasetWrapper
from ._cifar_10_h import CIFAR10H
from ._cifar_10_n import CIFAR10N
from ._cifar_100_n import CIFAR100N
from ._dopanim import Dopanim
from ._label_me import LabelMe
from ._music_genres import MusicGenres
from ._sentiment_polarity import SentimentPolarity


__all__ = [
    "MultiAnnotatorDataset",
    "SSLDatasetWrapper",
    "CIFAR10H",
    "CIFAR10N",
    "CIFAR100N",
    "Dopanim",
    "LabelMe",
    "MusicGenres",
    "SentimentPolarity",
]
