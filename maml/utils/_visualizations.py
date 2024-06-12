import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from skactiveml.utils import ExtLabelEncoder, rand_argmax
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


def plot_annotation_calibration_curves(likelihoods, y_true, classes, n_bins=10, annotators=None, path=None):
    """
    Plots the calibration curve, also referred to as reliability diagram, for the top-label prediction of the
    given likelihoods.

    Parameters
    ----------
    likelihoods : numpy.ndarray of shape (n_samples, n_annotators, n_classes)
        Normalized likelihoods per sample, annotator, and class. Missing likelihoods must be marked as `-1`.
    y_true : numpy.ndarray of shape (n_samples,)
        True class label per sample.
    classes : numpy.ndarray of shape (n_classes,)
        Possible class labels.
    n_bins : int, optional default=None
        Number of bins used for the calibration curve.
    annotators : numpy.ndarray of shape (n_annotators,), default=None
        If this array is not `None`, calibration curves are plotted per annotator where the array values are
        used as pseudonyms of the annotator names.
    path : str, default=None
        If `path` is a string, the path determines the location for saving the calibration curve(s).
    """

    def plot_calibration_curve(y_prob_max, is_correct, file_path):
        p_true, prob_pred = calibration_curve(y_true=is_correct, y_prob=y_prob_max, n_bins=n_bins)
        fig = plt.figure(figsize=(7, 5), dpi=100)  # plt.subplots_adjust(0, 0, 1, 1)
        ax_histogram = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Manually set the axes position
        ax_calibration = ax_histogram.twinx()
        ax_histogram.hist(y_prob_max, range=(0, 1), bins=n_bins, color="#800080ff")
        ax_calibration.plot(prob_pred, p_true, lw=3, color="#008080ff")
        ax_calibration.plot([0, 1], [0, 1], lw=3, color="#333333ff", ls="--")
        if file_path is not None:
            plt.savefig(file_path)
        plt.show()

    y_true = ExtLabelEncoder(classes=classes, missing_label=None).fit_transform(y_true)
    is_correct_all = []
    y_prob_max_all = []
    for a in range(likelihoods.shape[1]):
        y_prob = likelihoods[:, a]
        is_annotated = y_prob.sum(axis=-1) > 0
        y_prob = y_prob[is_annotated]
        y_pred = y_prob.argmax(axis=-1)
        is_correct = y_true[is_annotated] == y_pred
        y_prob_max = y_prob.max(axis=-1)
        is_correct_all.append(is_correct)
        y_prob_max_all.append(y_prob_max)
        if annotators is not None:
            file_path = os.path.join(path, f"calibration_curve_{annotators[a]}.pdf") if path is not None else None
            plot_calibration_curve(y_prob_max=y_prob_max, is_correct=is_correct, file_path=file_path)

    is_correct = np.concatenate(is_correct_all)
    y_prob_max = np.concatenate(y_prob_max_all)
    file_path = os.path.join(path, "calibration_curve_all.pdf") if path is not None else None
    plot_calibration_curve(y_prob_max=y_prob_max, is_correct=is_correct, file_path=file_path)


def plot_annotation_times_histograms(times, n_bins=50, annotators=None, path=None):
    """
    Plots the annotation times as histograms.

    Parameters
    ----------
    times : numpy.ndarray of shape (n_samples, n_annotators)
        Array of annotation times, where `times[x_idx, a_idx]` refers to the time annotator `a_idx` required to
        annotate sample `x_idx`.
    n_bins : int, optional default=None
        Number of bins used for the calibration curve.
    annotators : numpy.ndarray of shape (n_annotators,), default=None
        If this array is not `None`, histograms are plotted per annotator where the array values are
        used as pseudonyms of the annotator names.
    path : str, default=None
        If `path` is a string, the path determines the location for saving the annotation time histogram(s).
    """

    def plot_histogram(times, file_path):
        fig = plt.figure(figsize=(7, 5), dpi=100)
        ax_histogram = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax_histogram.hist(times, bins=n_bins, color="#800080ff")
        if file_path is not None:
            plt.savefig(file_path)
        plt.show()

    times_all = times.ravel()
    times_all = times_all[times_all > 0]
    file_path = os.path.join(path, "time_histogram_all.pdf") if path is not None else None
    plot_histogram(times=times_all, file_path=file_path)

    annotators = [] if annotators is None else annotators
    for a_idx, a in enumerate(annotators):
        times_a = times[:, a_idx]
        times_a = times_a[times_a > 0]
        file_path = os.path.join(path, f"time_histogram_{a}.pdf") if path is not None else None
        plot_histogram(times=times_a, file_path=file_path)


def plot_annotation_confusion_matrices(likelihoods, y_true, classes=None, annotators=None, path=None):
    """
    Plots the confusion matrices of the likelihoods' top-label predictions.

    Parameters
    ----------
    likelihoods : numpy.ndarray of shape (n_samples, n_annotators, n_classes)
        Normalized likelihoods per sample, annotator, and class. Missing likelihoods must be marked as `-1`.
    y_true : numpy.ndarray of shape (n_samples,)
        True class label per sample.
    classes : numpy.ndarray of shape (n_classes,)
        Possible class labels.
    annotators : numpy.ndarray of shape (n_annotators,), default=None
        If this array is not `None`, confusion matrices are plotted per annotator where the array values are
        used as pseudonyms of the annotator names.
    path : str, default=None
        If `path` is a string, the path determines the location for saving the confusion matrice(s).

    Returns
    -------
    cm_all : list
        The list of all annotators' confusion matrices, i.e., `cm_all[a_idx]` refers to the confusion matrix of
        annotator `a_idx`.
    """

    def plot_confusion_matrix(cm, file_path=None):
        # Define the colors
        colors = ["#007d7d99", "#7f007fff"]

        # Create a custom colormap
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=100)

        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(
            ax=ax,
            xticks_rotation="vertical",
            colorbar=False,
            include_values=False,
            cmap=cmap,
            im_kw={"alpha": 0.6},
        )
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(disp.im_, cax=cax)
        plt.xticks([])
        plt.yticks([])
        if file_path is not None:
            plt.savefig(file_path)
        plt.show()

    le = ExtLabelEncoder(classes=np.sort(classes), missing_label=None).fit(y_true)
    is_not_annotated = likelihoods.sum(axis=-1) == -15
    y_pred = rand_argmax(likelihoods, axis=-1, random_state=0)
    y_pred[is_not_annotated] = -1
    is_annotated = y_pred != -1
    y_pred = le.inverse_transform(y_pred)
    cm_all = []
    for a_idx in range(y_pred.shape[1]):
        cm = confusion_matrix(
            y_true=y_true[is_annotated[:, a_idx]],
            y_pred=y_pred[:, a_idx][is_annotated[:, a_idx]],
            normalize="true",
            labels=classes,
        )
        cm_all.append(cm)
        if annotators is not None:
            file_path = os.path.join(path, f"confusion_matrix_{annotators[a_idx]}.pdf") if path is not None else None
            plot_confusion_matrix(cm=cm, file_path=file_path)
    cm_sum = np.mean(cm_all, axis=0)
    file_path = os.path.join(path, "confusion_matrix_all.pdf") if path is not None else None
    plot_confusion_matrix(cm=cm_sum, file_path=file_path)

    return cm_all


def plot_annotator_metadata_correlation(annotator_metadata, cm_all, annotator_feature_names, path=None):
    """
    Plots the Spearman's rank correlation coefficient between actual annotators' accuracies and their individual
    metadata features.

    Parameters
    ----------
    annotator_metadata : numpy.ndarray (n_annotators, n_metadata_features)
        `annotator_metadata[a_idx, f_idx]` refers to the feature `f_idx` of annotator `a_idx`.
    cm_all : list
        The list of all annotators' confusion matrices, i.e., `cm_all[a_idx]` refers to the confusion matrix of
        annotator `a_idx`.
    annotator_feature_names : list
        The list of the annotator metadata feature names, i.e., `annotator_features_names[f_idx]` refers to the
        name of feature `f_idx`.
    path : str, default=None
        If `path` is a string, the path determines the location for saving the annotator metadata correlation plot.
    """

    annotator_metadata_df = pd.DataFrame(annotator_metadata, columns=annotator_feature_names)
    actual_accuracies = np.array([cm.diagonal().sum() / cm.sum() for cm in cm_all])
    annotator_metadata_df["actual_accuracies"] = actual_accuracies
    annotator_metadata_df = annotator_metadata_df[actual_accuracies < 0.9]

    # Compute the correlation matrix with the accuracy.
    correlation_matrix = annotator_metadata_df.corr(method="spearman")

    # Extract the correlation values for the accuracy.
    fig = plt.figure(figsize=(7, 5), dpi=100)  # plt.subplots_adjust(0, 0, 1, 1)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    correlation_with_accuracy = correlation_matrix["actual_accuracies"].drop("actual_accuracies")
    correlation_with_accuracy.plot.barh(ax=ax, color="#007f7f99")
    plt.ylabel("Annotator Metadata Features")
    plt.xlabel("Spearman Correlation Coefficient")
    if path is not None:
        file_path = os.path.join(path, "annotator_metadata_correlation.pdf")
        plt.savefig(file_path)
    plt.show()
