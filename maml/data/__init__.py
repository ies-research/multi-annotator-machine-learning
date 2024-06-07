from ._base import MultiAnnotatorDataset
from ._aloi_sim import ALOISim
from ._ag_news_sim import AGNewsSim
from ._cifar_10_h import CIFAR10H
from ._cifar_10_n import CIFAR10N
from ._cifar_100_n import CIFAR100N
from ._trec_6_sim import TREC6Sim
from ._label_me import LabelMe
from ._letter_sim import LetterSim
from ._music_genres import MusicGenres
from ._flowers_102_sim import Flowers102Sim
from ._dtd_sim import DTDSim

__all__ = [
    "MultiAnnotatorDataset",
    "ALOISim",
    "AGNewsSim",
    "CIFAR10H",
    "CIFAR10N",
    "CIFAR100N",
    "LabelMe",
    "LetterSim",
    "MusicGenres",
    "TREC6Sim",
    "Flowers102Sim",
    "DTDSim",
]
