import hydra
import sys
import warnings
import torch
import numpy as np
import pandas as pd

from hydra.utils import instantiate, get_class, to_absolute_path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar
from mlflow import get_experiment_by_name, set_tracking_uri, log_metric, start_run, create_experiment, MlflowException
from omegaconf.errors import ConfigAttributeError
from pprint import pprint
from skactiveml.utils import compute_vote_vectors, rand_argmax
from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader
from time import sleep

# TODO: In case of issues, set the absolute path to the directory of the mult-annotator-machine-learning project.
sys.path.append("../../")
warnings.filterwarnings("ignore")
set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")


# TODO: In case of issues, set the absolute path to the directory of hydra configuration files.
@hydra.main(config_path="../hydra_configs", config_name="experiment", version_base=None)
def evaluate(cfg):
    from maml.architectures import maml_net_params, gt_net
    from maml.data import SSLDatasetWrapper
    from maml.utils import log_params_from_omegaconf_dict

    # Setup experiment.
    set_tracking_uri(uri=f"file://{to_absolute_path(cfg.mlruns_path)}")
    exp = get_experiment_by_name(cfg.experiment_name)
    try:
        experiment_id = create_experiment(name=cfg.experiment_name) if exp is None else exp.experiment_id
    except MlflowException as e:
        print(e)
        for i in range(10):
            sleep(10)
            exp = get_experiment_by_name(cfg.experiment_name)
            experiment_id = exp.experiment_id
            if exp is not None:
                print("CONTINUE")
                break
        if exp is None:
            print("TERMINATE")

    # Print configuration.
    print("############ CONFIGURATION ############")
    pprint(dict(cfg))

    with start_run(experiment_id=experiment_id):
        # Log configuration.
        log_params_from_omegaconf_dict(cfg)

        # Set seed for deterministic results.
        seed_everything(cfg.seed, workers=True)

        # Load data.
        ds_train_cv = instantiate(
            cfg.data.class_definition,
            version="train",
            annotators=cfg.classifier.annotators,
            aggregation_method=cfg.classifier.aggregation_method,
        )
        ds_valid_cv = instantiate(cfg.data.class_definition, version="valid", annotators=cfg.classifier.annotators)
        ds_train_full = instantiate(
            cfg.data.class_definition,
            version="train",
            annotators=cfg.classifier.annotators,
            aggregation_method=cfg.classifier.aggregation_method,
            realistic_split=None,
        )
        if "n_annotations_per_sample" in cfg.data.class_definition:
            cfg.data.class_definition["n_annotations_per_sample"] = -1
        ds_test_full = instantiate(
            cfg.data.class_definition, version="test", annotators=cfg.classifier.annotators, realistic_split=None
        )

        # Set embedding dimension for AP architectures.
        try:
            embed_size = cfg.classifier.embed_size
        except ConfigAttributeError:
            embed_size = None

        # Build classifier architectures depending on the dataset.
        if cfg.data.lr_scheduler.class_definition is not None:
            lr_scheduler = get_class(cfg.data.lr_scheduler.class_definition)
        else:
            lr_scheduler = None

        # If desired, use SSL features.
        if cfg.ssl_model.name is not None:
            ssl_params_dict = cfg.ssl_model.params if cfg.ssl_model.params is not None else {}
            ssl_model, _, _ = gt_net(cfg.ssl_model.name, {"n_classes": ds_train_cv.get_n_classes(), **ssl_params_dict})
            ssl_model = ssl_model()
            ds_train_cv = instantiate(
                cfg.data.class_definition,
                annotators=cfg.classifier.annotators,
                transform=ds_test_full.transform if ds_test_full.transform else "auto",
                aggregation_method=cfg.classifier.aggregation_method,
            )
            ds_train_full = instantiate(
                cfg.data.class_definition,
                annotators=cfg.classifier.annotators,
                transform=ds_test_full.transform if ds_test_full.transform else "auto",
                aggregation_method=cfg.classifier.aggregation_method,
                realistic_split=None,
            )
            device = "cuda" if cfg.accelerator == "gpu" else "cpu"
            ds_train_cv = SSLDatasetWrapper(dataset=ds_train_cv, model=ssl_model, cache=True, device=device)
            ds_train_full = SSLDatasetWrapper(dataset=ds_train_full, model=ssl_model, cache=True, device=device)
            ds_valid_cv = SSLDatasetWrapper(dataset=ds_valid_cv, model=ssl_model, cache=True, device=device)
            ds_test_full = SSLDatasetWrapper(dataset=ds_test_full, model=ssl_model, cache=True, device=device)

        # Build data loaders.
        dl_train_cv = DataLoader(
            dataset=ds_train_cv,
            batch_size=cfg.data.train_batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            drop_last=True,
        )
        dl_train_full = DataLoader(
            dataset=ds_train_full,
            batch_size=cfg.data.train_batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            drop_last=True,
        )
        dl_valid_cv = DataLoader(
            dataset=ds_valid_cv, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers
        )
        dl_test_full = DataLoader(
            dataset=ds_test_full, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers
        )

        clf_dict = {}
        for state, dl in zip(["cv", "full"], [dl_train_cv, dl_train_full]):
            gt_params_dict = cfg.architecture.params if cfg.architecture.params is not None else {}
            params_dict = maml_net_params(
                gt_name=cfg.architecture.name,
                gt_params_dict={"n_classes": dl.dataset.get_n_classes(), **gt_params_dict},
                classifier_name=cfg.classifier.name,
                n_annotators=dl.dataset.get_n_annotators(),
                n_samples=len(dl.dataset),
                annotators=dl.dataset.get_annotators(),
                ap_confs=dl.dataset.ap_confs,
                annotation_mask=dl.dataset.get_annotation_matrix() != -1,
                classifier_specific=cfg.classifier.params,
                optimizer=get_class(cfg.data.optimizer.class_definition),
                optimizer_gt_dict=cfg.data.optimizer.gt_params,
                optimizer_ap_dict=cfg.data.optimizer.ap_params,
                lr_scheduler=lr_scheduler,
                lr_scheduler_dict=cfg.data.lr_scheduler.params,
                embed_size=embed_size,
            )
            clf_dict[state] = instantiate(cfg.classifier.class_definition, **params_dict)

            # Create callbacks for progressbar and checkpointing.
            bar = RichProgressBar()

            # Train multi-annotator classifier.
            trainer = Trainer(
                max_epochs=cfg.data.max_epochs,
                accelerator=cfg.accelerator,
                logger=False,
                callbacks=[bar],
                enable_checkpointing=False,
                deterministic="warn",
            )
            trainer.fit(model=clf_dict[state], train_dataloaders=dl)

        # Evaluate multi-annotator classifier after the last epoch.
        n_classes = ds_train_cv.get_n_classes()
        classes = np.arange(n_classes)
        dl_dict = {"cv": [("valid", dl_valid_cv)], "full": [("test", dl_test_full)]}
        device = "cuda" if cfg.accelerator == "gpu" else "cpu"
        for state, mdl in clf_dict.items():
            print(f"\n############ {state} ############")
            mdl.to(device)
            mdl.eval()
            for version, dl in dl_dict[state]:
                eval_scores_dict = {}
                normalizer_dict = {}
                # =================================== Collect data and predictions. ===================================
                for batch_idx, batch in enumerate(dl):
                    # Helper function for loss computation.
                    loss_name_list = ["zero_one_loss", "brier_score", "log_loss"]

                    def compute_losses(probas, targets, target_name, weights=None):
                        for loss_name in loss_name_list:
                            # Compute sample-wise losses.
                            if loss_name == "zero_one_loss":
                                preds = rand_argmax(probas, axis=-1, random_state=batch_idx)
                                preds = np.eye(n_classes)[preds]
                                loss = 1 - np.einsum("ic,ic->i", targets, preds)
                            elif loss_name == "brier_score":
                                loss = ((targets - probas) ** 2).sum(axis=-1)
                            elif loss_name == "log_loss":
                                loss = (-targets * np.log(probas + np.finfo(np.float32).eps)).sum(axis=-1)
                            else:
                                raise NotImplementedError(f"`{loss_name}` is not implemented.")

                            # Sum losses.
                            if weights is not None:
                                loss = (loss * weights).sum()
                                normalizer = weights.sum()
                            else:
                                loss = loss.sum()
                                normalizer = len(targets)

                            # Log losses.
                            loss_key = f"{target_name}_{loss_name}_{state}_{version}"
                            eval_scores_dict[loss_key] = eval_scores_dict.get(loss_key, 0) + loss
                            normalizer_dict[loss_key] = normalizer_dict.get(loss_key, 0) + normalizer

                    # Compute predictions.
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred_dict = mdl.predict_step(batch=batch, batch_idx=batch_idx)
                    batch = {k: v.cpu().numpy() for k, v in batch.items()}
                    pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

                    # Compute predictions and targets.
                    p_class = pred_dict["p_class"]
                    p_annot = pred_dict.get("p_annot", None)
                    targets_dict = {"class_true": np.eye(n_classes)[batch["y"]]}
                    weights_dict = {}
                    z = batch.get("z", None)
                    if z is not None:
                        # Mask out missing annotations.
                        n_annotators = z.shape[-1]
                        is_labeled = z != -1
                        z_ravel = z.ravel()
                        is_labeled_ravel = is_labeled.ravel()
                        z_ravel = z_ravel[is_labeled_ravel]
                        z_ravel_one_hot = np.eye(n_classes)[z_ravel]
                        if p_annot is not None:
                            p_annot = p_annot.reshape((-1, n_classes))[is_labeled_ravel]

                        # Storage for estimated probabilities.
                        p_dict = {}

                        # (Soft) majority voting and noisy annotations with uniform weights.
                        p_unif = compute_vote_vectors(y=z, missing_label=-1, classes=classes)
                        p_unif /= p_unif.sum(axis=-1, keepdims=True)
                        p_dict["unif"] = p_unif

                        # Get annotator's performances.
                        p_perf = pred_dict.get("p_perf", None)
                        if p_perf is not None:
                            # Compute probabilities with log-odds as weights.
                            p_perf_clipped = np.clip(p_perf, 1e-3, 1 - 1e-3)
                            log_odds = np.log((n_classes - 1) * p_perf_clipped / (1 - p_perf_clipped))
                            scores = compute_vote_vectors(y=z, classes=classes, w=log_odds, missing_label=-1)
                            scores_stable = scores - scores.max(axis=1, keepdims=True)
                            for temp_scaling in [1, 1.5, 2, 2.5, 3]:
                                exp_scores = np.exp(scores_stable / temp_scaling)
                                p_dict[f"log_odds_{temp_scaling}"] = exp_scores / exp_scores.sum(axis=1, keepdims=True)

                            # Compute probabilities with linear weights.
                            scores = compute_vote_vectors(y=z, w=p_perf, missing_label=-1, classes=classes)
                            p_dict["linear"] = scores / scores.sum(axis=-1, keepdims=True)

                        # Compute estimated targets.
                        for p_type, p_est in p_dict.items():
                            y_est = rand_argmax(p_est, axis=-1, random_state=batch_idx)
                            targets_dict[f"class_mv_{p_type}"] = np.eye(n_classes)[y_est]
                            weights_dict[f"class_mv_{p_type}"] = p_est.max(axis=-1)
                            targets_dict[f"class_smv_{p_type}"] = p_est
                            if p_annot is not None:
                                p_est_ravel = p_est.repeat(repeats=n_annotators, axis=0)[is_labeled_ravel]
                                targets_dict[f"annot_{p_type}"] = z_ravel_one_hot
                                weights_dict[f"annot_{p_type}"] = (z_ravel_one_hot * p_est_ravel).sum(axis=-1)

                    # Evaluate predictions.
                    for k, v in targets_dict.items():
                        # Compute scores without sample weights.
                        probas = p_class if k.startswith("class") else p_annot
                        compute_losses(probas=probas, targets=v, target_name=k)
                        if k.startswith("class") and version == "valid":
                            compute_losses(probas=v, targets=targets_dict["class_true"], target_name=f"{k}_quality")

                        # Compute scores with samples weights.
                        w = weights_dict.get(k, None)
                        if w is not None:
                            compute_losses(probas=probas, targets=v, target_name=f"{k}_weighted", weights=w)
                            if k.startswith("class") and version == "valid":
                                compute_losses(
                                    probas=v,
                                    targets=targets_dict["class_true"],
                                    target_name=f"{k}_weighted_quality",
                                    weights=w,
                                )

                # Normalize evaluation scores.
                for k, v in eval_scores_dict.items():
                    eval_scores_dict[k] = [eval_scores_dict[k] / normalizer_dict[k]]
                    log_metric(k, eval_scores_dict[k][0])
                eval_df = pd.DataFrame(eval_scores_dict).T
                eval_df.columns = ["value"]
                eval_df.index.name = version
                for loss_name in loss_name_list:
                    is_loss = [l for l in eval_df.index if loss_name in l]
                    print(f"{eval_df.loc[is_loss].to_markdown(tablefmt='github', floatfmt='.4f')}")
                    print("\n")


if __name__ == "__main__":
    evaluate()
