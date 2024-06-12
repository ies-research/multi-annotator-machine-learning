import numpy as np
import os
import pandas as pd
import torch
import json

from PIL import Image
from skactiveml.utils import ExtLabelEncoder, rand_argmax
from sklearn.preprocessing import StandardScaler
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
from numpy.typing import ArrayLike
from typing import Literal, Optional

from ._base import MultiAnnotatorDataset, AGGREGATION_METHODS, TRANSFORMS, VERSIONS


class Dopanim(MultiAnnotatorDataset):
    """Dopanim

    The Dopanim [1] dataset features about 15,750 animal images of 15 classes, organized into four groups of
    doppelganger animals and collected together with ground truth labels from iNaturalist. For approximately 10,500 of
    these images, 20 humans provided over 52,000 annotations with an accuracy of circa 67%.

    Parameters
    ----------
    root : str
        Path to the root directory, where the ata is located.
    version : "train" or "valid" or "test", default="train"
        Defines the version (split) of the dataset.
    download : bool, default=False
        Flag whether the dataset will be downloaded.
    annotators : None or "index" or "one-hot" or "metadata"
        Defines the representation of the annotators as either indices, one-hot encoded vectors, real metadata, or
        `None`.
    aggregation_method : str, default=None
        Supported methods are majority voting (`aggregation_method="majority_vote") and returning the true class
        labels (`aggregation_method="ground-truth"). In the case of `aggregation_method=None`, `None` is returned
        as aggregated annotations.
    transform : "auto" or torch.nn.Module, default="auto"
        Transforms for the samples, where "auto" used pre-defined transforms fitting the respective version.
    variant :"worst-1" or "worst-2" or "worst-3" or "worst-4" or "worst-v" or "rand-1" or "rand-2" or "rand-3" or
    "rand-4" or "rand-v" or "full"
        Defines subsets of annotations to reflect different learning scenarios.
    annotation_type : "class-labels" or "probabilities", default="class-labels",
        Defines which type of annotations is used.

    References
    ----------
    [1] Herde, M., Huseljic, D., Rauch, L., & Sick, B. (2024). dopanim: A Dataset of Doppelganger Animals with Noisy
        Annotations from Multiple Humans [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11479590
    """

    base_folder = "dopanim_zenodo_download"
    url = "https://zenodo.org/api/records/11479590/files-archive"
    filename = "11479590.zip"
    image_dir = "images"
    classes = np.array(
        [
            "German Yellowjacket",
            "European Paper Wasp",
            "Yellow-legged Hornet",
            "European Hornet",
            "Brown Hare",
            "Black-tailed Jackrabbit",
            "Marsh Rabbit",
            "Desert Cottontail",
            "European Rabbit",
            "Eurasian Red Squirrel",
            "American Red Squirrel",
            "Douglas' Squirrel",
            "Cheetah",
            "Jaguar",
            "Leopard",
        ],
        dtype=object,
    )
    annotators = np.array(
        [
            "digital-dragon",
            "pixel-pioneer",
            "ocean-oracle",
            "starry-scribe",
            "sunlit-sorcerer",
            "emerald-empath",
            "sapphire-sphinx",
            "echo-eclipse",
            "lunar-lynx",
            "neon-ninja",
            "quantum-quokka",
            "velvet-voyager",
            "radiant-raven",
            "dreamy-drifter",
            "azure-artist",
            "twilight-traveler",
            "galactic-gardener",
            "cosmic-wanderer",
            "frosty-phoenix",
            "mystic-merlin",
        ],
        dtype=object,
    )
    variants = np.array(
        [
            "full",
            "worst-var",
            "rand-var",
            "worst-1",
            "worst-2",
            "worst-3",
            "worst-4",
            "rand-1",
            "rand-2",
            "rand-3",
            "rand-4",
        ],
        dtype=object,
    )

    def __init__(
        self,
        root: str,
        version: VERSIONS = "train",
        download: bool = False,
        annotators: Optional[Literal["one-hot", "index", "metadata"]] = None,
        aggregation_method: AGGREGATION_METHODS = None,
        transform: TRANSFORMS = "auto",
        variant: str = "worst-1",
        annotation_type: Literal["class-labels", "probabilities"] = "class-labels",
    ):
        # Download data.
        self.folder = os.path.join(root, Dopanim.base_folder)
        if download:
            download_and_extract_archive(Dopanim.url, root, filename=Dopanim.filename, extract_root=self.folder)
            version_filename = os.path.join(self.folder, f"{version}.zip")
            download_and_extract_archive(Dopanim.url, root, filename=version_filename, extract_root=self.folder)

        # Set dataset parameters.
        self.variant = variant
        self.annotation_type = annotation_type

        # Check availability of data.
        is_available = os.path.exists(self.folder)
        if not is_available:
            raise RuntimeError("Dataset not found. You can use `download=True` to download it.")

        # Load annotation file.
        if version not in ["train", "valid", "test"]:
            raise ValueError("`version` must be in `['train', 'valid', 'test']`.")
        self.img_folder = os.path.join(self.folder, version)

        # Load and prepare true labels as tensor.
        self.y_orig, self.observation_ids = self.load_true_class_labels(version=version)
        self.le = ExtLabelEncoder(missing_label=None, classes=Dopanim.classes).fit(self.y_orig)
        self.y = self.le.transform(self.y_orig)

        # Load and prepare annotations as tensor.
        self.z = self.load_annotations() if version == "train" else None

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

        # Transform to tensors.
        self.y = torch.from_numpy(self.y)
        self.z = torch.from_numpy(self.z) if self.z is not None else None

        # Load and prepare annotator features as tensor if `annotators` is not `None`.
        if annotators == "metadata":
            z = self.load_annotations() if self.z is None else self.z
            self.a, _ = self.load_annotator_metadata(
                classes=self.le.classes_,
                annotators=Dopanim.annotators,
                is_not_annotated=z == -1,
            )
            self.a = torch.from_numpy(StandardScaler().fit_transform(self.a)).float()
        else:
            self.a = self.prepare_annotator_features(annotators=annotators, n_annotators=self.get_n_annotators())

        # Aggregate annotations if `aggregation_method` is not `None`.
        self.z_agg = self.aggregate_annotations(z=self.z, y=self.y, aggregation_method=aggregation_method)

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
        return len(Dopanim.classes)

    def get_n_annotators(self):
        """
        Returns
        -------
        n_annotators : int
            Number of annotators.
        """
        return len(Dopanim.annotators)

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
        x = Image.open(os.path.join(self.img_folder, self.y_orig[idx], f"{self.observation_ids[idx]}.jpeg"))
        x = x.convert("RGB")
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

    def load_annotations(self):
        """
        Loads the annotations of the given variant and annotation type.

        Returns
        -------
        z : np.ndarray of shape (n_samples, n_annotators) or (n_samples, n_annotators, n_classes)
            Observed annotations.
        """
        y_true_train, observation_ids_train = self.load_true_class_labels(version="train")
        y_true_train = self.le.transform(y_true_train)
        likelihoods = self.load_likelihoods(
            observation_ids=observation_ids_train,
            classes=self.le.classes_,
            annotators=Dopanim.annotators,
            normalize=True,
        )
        is_not_annotated = np.any(likelihoods == -1, axis=-1)
        class_labels = rand_argmax(likelihoods, axis=-1, random_state=0)
        class_labels[is_not_annotated] = -1
        if self.annotation_type == "probabilities":
            z = likelihoods
        elif self.annotation_type == "class-labels":
            z = class_labels
        else:
            raise ValueError(
                f"`annotation_type` must be in ['class-labels', 'probabilities'], got '{self.annotation_type}' instead."
            )
        if self.variant in ["worst-1", "worst-2", "worst-3", "worst-4"]:
            n_annotators_per_sample = int(self.variant.split("-")[-1])
            is_false = np.full_like(class_labels, fill_value=0.0, dtype=float)
            is_false += (y_true_train[:, None] != class_labels).astype(float)
            is_false -= 2 * is_not_annotated.astype(float)
            is_not_worst = np.full_like(is_false, fill_value=True, dtype=bool)
            random_floats = np.random.RandomState(n_annotators_per_sample).rand(*is_false.shape)
            worst_indices = np.argsort(-(is_false + random_floats), axis=-1)[:, :n_annotators_per_sample]
            for c in range(n_annotators_per_sample):
                is_not_worst[np.arange(len(is_not_worst)), worst_indices[:, c]] = False
            z[is_not_worst] = -1
        elif self.variant in ["rand-1", "rand-2", "rand-3", "rand-4"]:
            n_annotators_per_sample = int(self.variant.split("-")[-1])
            is_annotated = (~is_not_annotated).astype(float)
            is_not_selected = np.full_like(is_annotated, fill_value=True, dtype=bool)
            random_floats = np.random.RandomState(n_annotators_per_sample + 4).rand(*is_annotated.shape)
            random_indices = np.argsort(-(is_annotated + random_floats), axis=-1)[:, :n_annotators_per_sample]
            for c in range(n_annotators_per_sample):
                is_not_selected[np.arange(len(is_annotated)), random_indices[:, c]] = False
            z[is_not_selected] = -1
        elif self.variant in ["rand-var", "worst-var"]:
            random_state = np.random.RandomState(0)
            for i in range(len(is_not_annotated)):
                # Get the indices of ones in the current row
                annotated_indices = np.where(is_not_annotated[i] == False)[0]

                # Determine the size of the subset to set to zero
                subset_size = random_state.randint(0, len(annotated_indices))

                # Select worst indices to set to zero
                if self.variant == "worst-var":
                    is_false = class_labels[i][annotated_indices] == y_true_train[i]
                    random_floats = random_state.rand(*is_false.shape)
                    worst_indices = np.argsort(-(is_false + random_floats), axis=-1)[:subset_size]
                    indices_to_true = annotated_indices[worst_indices]
                else:
                    # Randomly select indices to set to zero
                    indices_to_true = random_state.choice(annotated_indices, size=subset_size, replace=False)

                # Set the selected indices to zero
                is_not_annotated[i, indices_to_true] = True
            z[is_not_annotated] = -1
        elif self.variant == "full":
            pass
        else:
            raise ValueError(f"`variant` must be in {Dopanim.variants}, got '{self.variant}' instead.")
        return z

    def load_true_class_labels(self, version: VERSIONS = "train"):
        """
        Loads the true class of the given version.

        Parameters
        ----------
        version : "train" or "valid" or "test"
            Version (split) of the dataset.

        Returns
        -------
        z : np.ndarray of shape (n_samples,)
            True class labels.
        """
        with open(os.path.join(self.folder, "task_data.json")) as task_file:
            task_data = json.load(task_file)
        y_true_list = []
        observation_id_list = []
        for observation_id, observation_dict in task_data.items():
            if observation_dict["split"] == version:
                y_true_list.append(observation_dict["taxon_name"])
                observation_id_list.append(observation_id)
        return np.array(y_true_list, dtype=object), np.array(observation_id_list, dtype=int)

    def load_likelihoods(
        self,
        observation_ids: ArrayLike,
        annotators: Optional[ArrayLike] = None,
        classes: Optional[ArrayLike] = None,
        normalize: bool = True,
    ):
        """
        Loads the likelihoods for the given observation IDs.

        Parameters
        ----------
        observation_ids : array-like of shape (n_obs_ids,)
            Observation IDs whose likelihoods are loaded.
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose likelihoods are loaded.
        classes : array-like of shape (n_classes,), default=None
            Defines the order of the class labels.
        normalize : bool, default=True
            Flag whether likelihoods are normalized.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_annotators, n_classes)
            Observed likelihoods.
        """
        with open(os.path.join(self.folder, "annotation_data.json")) as task_file:
            annotation_data = json.load(task_file)
        annotators = Dopanim.annotators if annotators is None else annotators
        classes = Dopanim.classes if classes is None else classes
        likelihoods = np.full(
            (len(observation_ids), len(annotators), len(classes)),
            fill_value=-1,
            dtype=float,
        )
        for annotation_id, annotation_dict in annotation_data.items():
            obs_idx = np.where(annotation_dict["observation_id"] == observation_ids)[0]
            annot_idx = np.where(annotation_dict["annotator_id"] == annotators)[0]
            for class_name, likelihood_value in annotation_dict["likelihoods"].items():
                cls_idx = np.where(class_name == classes)[0]
                likelihoods[obs_idx, annot_idx, cls_idx] = float(likelihood_value)
            if normalize and likelihoods[obs_idx, annot_idx].sum() > 0:
                likelihoods[obs_idx, annot_idx] = (
                    likelihoods[obs_idx, annot_idx] / likelihoods[obs_idx, annot_idx].sum()
                )
        return likelihoods

    def load_annotation_consistencies(
        self,
        observation_ids: ArrayLike,
        annotators: Optional[ArrayLike] = None,
        is_not_annotated: Optional[ArrayLike] = None,
    ):
        """
        Loads the annotation consistencies per annotator.

        Parameters
        ----------
        observation_ids : array-like of shape (n_obs_ids,)
            Observation IDs for which the annotation consistencies are loaded.
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose annotation consistencies are loaded.
        is_not_annotated : array-like of shape (n_samples, n_annotators), default=None
            A boolean mask indicating which sample is annotated by which annotator. If `ìs_not_annotated=None`, the
            missing annotations are computed for the `full` variant.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_annotators,)
            Observed likelihoods.
        """
        with open(os.path.join(self.folder, "annotation_data.json")) as task_file:
            annotation_data = json.load(task_file)
        num_inconsistencies = np.zeros(len(annotators), dtype=float)
        num_duplicates = np.zeros(len(annotators), dtype=float)

        y_pred = np.full((len(observation_ids), len(annotators)), fill_value=None)
        for annotation_id, annotation_dict in annotation_data.items():
            obs_idx = np.where(annotation_dict["observation_id"] == observation_ids)[0][0]
            annot_idx = np.where(annotation_dict["annotator_id"] == annotators)[0][0]
            if is_not_annotated is not None and is_not_annotated[obs_idx, annot_idx]:
                continue
            max_likelihood = -1
            y_pred_new = None
            for class_name, likelihood_value in annotation_dict["likelihoods"].items():
                if max_likelihood < likelihood_value:
                    y_pred_new = class_name
                    max_likelihood = likelihood_value
            if y_pred_new is not None:
                if y_pred[obs_idx, annot_idx] is not None:
                    num_duplicates[annot_idx] += 1
                    num_inconsistencies[annot_idx] += int(y_pred_new != y_pred[obs_idx, annot_idx])
                y_pred[obs_idx, annot_idx] = y_pred_new
        annotation_inconsistencies = num_inconsistencies / num_duplicates
        annotation_consistencies = 1 - np.nan_to_num(annotation_inconsistencies, nan=0)
        return annotation_consistencies

    def load_annotation_times(self, observation_ids: ArrayLike, annotators: Optional[ArrayLike] = None):
        """
        Loads the annotation times per sample and annotator.

        Parameters
        ----------
        observation_ids : array-like of shape (n_obs_ids,)
            Observation IDs for which annotation times are loaded.
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose annotation times are loaded.

        Returns
        -------
        annotation_times : np.ndarray of shape (n_samples, n_annotators)
            Observed annotation times.
        """
        with open(os.path.join(self.folder, "annotation_data.json")) as task_file:
            annotation_data = json.load(task_file)
        annotation_times = np.full((len(observation_ids), len(annotators)), fill_value=-1, dtype=float)
        for annotation_id, annotation_dict in annotation_data.items():
            obs_idx = np.where(annotation_dict["observation_id"] == observation_ids)[0][0]
            annot_idx = np.where(annotation_dict["annotator_id"] == annotators)[0][0]
            annotation_times[obs_idx, annot_idx] = annotation_dict["annotation_time"]

        for annot_idx in range(len(annotators)):
            times_annot_idx = annotation_times[:, annot_idx]
            is_annotated = times_annot_idx > 0
            is_outlier = times_annot_idx > np.quantile(times_annot_idx[is_annotated], q=0.95)
            random_state = np.random.RandomState(0)
            times_annot_idx[is_outlier] = random_state.uniform(
                low=np.quantile(times_annot_idx[is_annotated], q=0.1),
                high=np.quantile(times_annot_idx[is_annotated], q=0.9),
                size=np.sum(is_outlier),
            )
            annotation_times[:, annot_idx] = times_annot_idx
        return annotation_times

    def load_annotator_metadata(
        self, annotators: Optional[ArrayLike] = None, is_not_annotated: Optional[ArrayLike] = None
    ):
        """
        Loads the metadata per annotator.

        Parameters
        ----------
        annotators : array-like of shape (n_annotators,), default=None
            Names of the annotators whose metadata are loaded.
        is_not_annotated : array-like of shape (n_samples, n_annotators), default=None
            A boolean mask indicating which sample is annotated by which annotator. If `ìs_not_annotated=None`, the
            missing annotations are computed for the `full` variant.

        Returns
        -------
        annotator_metadata_values : np.ndarray of shape (n_annotators, n_metadata_features)
            Observed metadata.
        annotator_metadata_names : np.ndarray of shape (n_metadata_features,)
            Feature names of the metadata.
        """
        with open(os.path.join(self.folder, "annotator_metadata.json")) as task_file:
            annotator_metadata = json.load(task_file)
        # Encoder for questions in the pre-questionnaire.
        pre_encoder = {
            # Encoder of self-assessments questions.
            "pre_interest_choice": {
                "Very low": 0,
                "Below average": 1,
                "Average": 2,
                "Above average": 3,
                "Very high": 4,
            },
            "pre_knowledge_choice": {
                "Very low": 0,
                "Below average": 1,
                "Average": 2,
                "Above average": 3,
                "Very high": 4,
            },
            # Encoder of technical questions.
            "pre_oldest_animal_choice": {
                "African elephant": 0,
                "Galapagos tortoise": 0,
                "Blue whale": 0,
                "Greenland shark": 1,
                "Aldabra giant tortoise": 0,
            },
            "pre_mammal_migration_choice": {
                "Humpback whale": 0,
                "Bactrian camel": 0,
                "African elephant": 0,
                "Caribou": 1,
            },
            "pre_big_cats_choice": {"Tiger": 0, "Jaguar": 0, "Leopard": 1, "Cheetah": 0},
            "pre_hares_choice": {
                "Rabbits are born blind and hairless, hares are born with fur and can see.": 1,
                "Hares are born blind and hairless, rabbits are born with fur and can see.": 0,
                "Both are born fully furred and with open eyes.": 0,
                "Both are born blind and hairless.": 0,
            },
            "pre_squirrels_choice": {
                "Squirrel tails help with balance.": 1,
                "Chipmunks are members of the squirrel family.": 0,
                "Squirrels' teeth do not stop growing.": 0,
                "Squirrels can remember the exact location of every nut they bury.": 0,
            },
            "pre_insects_choice": {
                "Wasps have eight legs.": 0,
                "Wasps die after a sting.": 0,
                "Acids in the wasps venom can lead to allergic reactions.": 0,
                "Hornets are one of the most commonly known wasp.": 1,
            },
        }

        # Encoder for questions in the pre-questionnaire.
        post_encoder = {
            # Encoder of general self-assessment-questions.
            "post_likelihood_choice": {
                "Very poor": 0,
                "Poor": 1,
                "Acceptable": 2,
                "Good": 3,
                "Very good": 4,
            },
            "post_concentration_choice": {
                "Very unconcentrated": 0,
                "Somewhat unconcentrated": 1,
                "Neither unconcentrated nor concentrated": 2,
                "Somewhat concentrated": 3,
                "Very concentrated": 4,
            },
            "post_tutorial_choice": {
                "Never": 0,
                "Rarely": 1,
                "Sometimes": 2,
                "Often": 3,
                "Very often": 4,
            },
            "post_motivation_choice": {
                "Very unmotivated": 0,
                "Somewhat unmotivated": 1,
                "Neither unmotivated nor motivated": 2,
                "Somewhat motivated": 3,
                "Very motivated": 4,
            },
            # Encoder of big cats related self-assessment-questions.
            "post_jaguar_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_leopard_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_cheetah_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            # Encoder of hares & rabbits related self-assessment-questions.
            "post_brown_hare_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_black_tailed_jackrabbit_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_marsh_rabbit_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_european_rabbit_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_desert_cottontail_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            # Encoder of squirrels related self-assessment-questions.
            "post_douglas_squirrel_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_american_red_squirrel_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_eurasian_red_squirrel_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            # Encoder of insects related self-assessment-questions.
            "post_asian_hornet_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_european_hornet_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_european_paper_wasp_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
            "post_german_yellowjacket_choice": {
                "Very easy": 4,
                "Somewhat easy": 3,
                "Neither easy nor difficult": 2,
                "Somewhat difficult": 1,
                "Very difficult": 0,
            },
        }

        pre_self_assessment_questions = [
            "pre_interest_choice",
            "pre_knowledge_choice",
        ]
        pre_technical_questions = [
            "pre_hares_choice",
            "pre_insects_choice",
            "pre_big_cats_choice",
            "pre_squirrels_choice",
            "pre_oldest_animal_choice",
            "pre_mammal_migration_choice",
        ]
        post_self_assessment_questions = [
            "post_likelihood_choice",
            "post_concentration_choice",
            "post_tutorial_choice",
            "post_motivation_choice",
            "post_jaguar_choice",
            "post_leopard_choice",
            "post_cheetah_choice",
            "post_brown_hare_choice",
            "post_black_tailed_jackrabbit_choice",
            "post_marsh_rabbit_choice",
            "post_european_rabbit_choice",
            "post_desert_cottontail_choice",
            "post_douglas_squirrel_choice",
            "post_american_red_squirrel_choice",
            "post_eurasian_red_squirrel_choice",
            "post_asian_hornet_choice",
            "post_european_hornet_choice",
            "post_european_paper_wasp_choice",
            "post_german_yellowjacket_choice",
        ]

        _, observation_ids = self.load_true_class_labels(version="train")

        # Compute mean maximum likelihood per annotator.
        likelihoods = self.load_likelihoods(
            observation_ids=observation_ids,
            annotators=annotators,
            classes=Dopanim.classes,
        )
        likelihoods[likelihoods.sum(axis=-1) == -15] = np.nan
        if is_not_annotated is not None:
            likelihoods[is_not_annotated] = np.nan
        mean_max_likelihoods = np.nanmean(likelihoods.max(axis=2), axis=0)

        # Compute mean annotation time.
        annotation_times = self.load_annotation_times(observation_ids=observation_ids, annotators=annotators)
        annotation_times[annotation_times == -1] = np.nan
        if is_not_annotated is not None:
            annotation_times[is_not_annotated] = np.nan
        mean_annotation_times = np.nanmean(annotation_times, axis=0)

        # Compute annotation inconsistencies.
        annotation_consistencies = self.load_annotation_consistencies(
            observation_ids=observation_ids,
            annotators=annotators,
            is_not_annotated=is_not_annotated,
        )

        preprocessed_annotator_metadata = {}
        for annot_idx, annot_name in enumerate(annotators):
            annot_metadata = {}

            pre_tech_acc = 0

            for key in annotator_metadata[annot_name]:
                answer = annotator_metadata[annot_name][key]

                # pre interest
                if key in pre_self_assessment_questions:
                    encoded_answer = pre_encoder[key][answer]
                    annot_metadata[key] = encoded_answer
                # pre tech
                elif key in pre_technical_questions:
                    encoded_answer = pre_encoder[key][answer]
                    pre_tech_acc += encoded_answer
                # post
                elif key in post_self_assessment_questions:
                    encoded_answer = post_encoder[key][answer]
                    annot_metadata[key] = encoded_answer
                elif key in [
                    "pre_time",
                    "post_time",
                    "post_estimated_accuracy",
                ] or key.endswith("tutorial"):
                    annot_metadata[key] = answer
            annot_metadata["pre_tech_accuracy"] = pre_tech_acc / len(pre_technical_questions)
            annot_metadata["mean_max_likelihood"] = mean_max_likelihoods[annot_idx]
            annot_metadata["mean_annotation_time"] = mean_annotation_times[annot_idx]
            annot_metadata["annotation_consistency"] = annotation_consistencies[annot_idx]
            preprocessed_annotator_metadata[annot_name] = annot_metadata
        annotator_metadata = pd.DataFrame(preprocessed_annotator_metadata).T

        # Compute new feature.
        tutorial_columns = [
            "basic_tutorial",
            "big_cats_tutorial",
            "hares_rabbits_tutorial",
            "insects_tutorial",
            "squirrels_tutorial",
        ]
        annotator_metadata["num_tutorials"] = annotator_metadata[tutorial_columns].mean(axis=1)
        annotator_metadata = annotator_metadata.drop(columns=tutorial_columns)

        # Compute new feature.
        class_columns = [
            "post_jaguar_choice",
            "post_leopard_choice",
            "post_cheetah_choice",
            "post_brown_hare_choice",
            "post_black_tailed_jackrabbit_choice",
            "post_marsh_rabbit_choice",
            "post_european_rabbit_choice",
            "post_desert_cottontail_choice",
            "post_douglas_squirrel_choice",
            "post_american_red_squirrel_choice",
            "post_eurasian_red_squirrel_choice",
            "post_asian_hornet_choice",
            "post_european_hornet_choice",
            "post_european_paper_wasp_choice",
            "post_german_yellowjacket_choice",
        ]
        annotator_metadata["class_confidence"] = annotator_metadata[class_columns].mean(axis=1)
        annotator_metadata = annotator_metadata.drop(columns=class_columns)

        return annotator_metadata.values, annotator_metadata.columns
