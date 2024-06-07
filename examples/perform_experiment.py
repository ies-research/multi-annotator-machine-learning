import os
import hydra
import sys
import warnings
import mlflow
import atexit
import signal
import torch

from hydra.utils import instantiate, get_class, to_absolute_path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from mlflow import get_experiment_by_name, set_tracking_uri, log_metric, start_run, create_experiment
from omegaconf.errors import ConfigAttributeError
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader

sys.path.append("../")
warnings.filterwarnings("ignore")
set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(config_path="../conf", config_name="experiment", version_base=None)
def evaluate(cfg):
    from maml.architectures import maml_net_params
    from maml.utils import log_params_from_omegaconf_dict

    # Setup experiment.
    set_tracking_uri(uri=f"file://{to_absolute_path(cfg.mlruns_path)}")
    exp = get_experiment_by_name(cfg.experiment_name)
    experiment_id = create_experiment(name=cfg.experiment_name) if exp is None else exp.experiment_id

    print(cfg.data.class_definition._target_)
    print(cfg.classifier.name)

    with start_run(experiment_id=experiment_id):
        # Get path to artifacts and ensure that artifact is deleted after program termination or cancellation.
        artifacts_path = mlflow.active_run().info.artifact_uri.split("file://")[1]
        def cleanup_function():
            if os.path.exists(artifacts_path):
                try:
                    os.remove(os.path.join(artifacts_path, "best.ckpt"))
                except FileNotFoundError:
                    pass
        atexit.register(cleanup_function)
        def signal_handler(sig, frame):
            cleanup_function()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        # Log configuration.
        log_params_from_omegaconf_dict(cfg)

        # Set seed for deterministic results.
        seed_everything(cfg.seed, workers=True)

        # Load data.
        ds_train = instantiate(
            cfg.data.class_definition,
            version="train",
            annotators=cfg.classifier.annotators,
            aggregation_method=cfg.classifier.aggregation_method,
        )
        ds_valid = instantiate(cfg.data.class_definition, version="valid")
        if "n_annotations_per_sample" in cfg.data.class_definition:
            cfg.data.class_definition["n_annotations_per_sample"] = -1
        ds_test = instantiate(cfg.data.class_definition, version="test", annotators=cfg.classifier.annotators)
        ds_train_eval = instantiate(
            cfg.data.class_definition,
            version="train",
            annotators=cfg.classifier.annotators,
            transform=ds_test.transform if ds_test.transform else "auto",
        )

        # Build data loaders.
        dl_train = DataLoader(
            dataset=ds_train, batch_size=cfg.data.train_batch_size, num_workers=cfg.data.num_workers, shuffle=True
        )
        dl_train_eval = DataLoader(
            dataset=ds_train_eval, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers
        )
        dl_valid = DataLoader(dataset=ds_valid, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers)
        dl_test = DataLoader(dataset=ds_test, batch_size=cfg.data.eval_batch_size, num_workers=cfg.data.num_workers)

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
        gt_params_dict = cfg.architecture.params if cfg.architecture.params is not None else {}
        params_dict = maml_net_params(
            gt_name=cfg.architecture.name,
            gt_params_dict={"n_classes": ds_train.get_n_classes(), **gt_params_dict},
            classifier_name=cfg.classifier.name,
            n_annotators=ds_train.get_n_annotators(),
            annotators=ds_train.get_annotators(),
            classifier_specific=cfg.classifier.params,
            optimizer=get_class(cfg.data.optimizer.class_definition),
            optimizer_gt_dict=cfg.data.optimizer.gt_params,
            optimizer_ap_dict=cfg.data.optimizer.ap_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_dict=cfg.data.lr_scheduler.params,
            embed_size=embed_size,
        )
        clf = instantiate(cfg.classifier.class_definition, **params_dict)

        # Create callbacks for progressbar and checkpointing.
        bar = RichProgressBar()
        checkpoint = ModelCheckpoint(
            monitor="gt_val_acc",
            dirpath=artifacts_path,
            filename="best",
            mode="max",
            save_top_k=1,
            save_last=False,
        )

        # Train multi-annotator classifier.
        trainer = Trainer(
            max_epochs=cfg.data.max_epochs,
            accelerator=cfg.accelerator,
            logger=False,
            callbacks=[bar, checkpoint],
            deterministic="warn",
        )
        trainer.fit(model=clf, train_dataloaders=dl_train, val_dataloaders=dl_valid)

        # Evaluate multi-annotator classifier after the last epoch.
        dl_list = [("train", dl_train_eval), ("valid", dl_valid), ("test", dl_test)]
        device = "cuda" if cfg.accelerator == "gpu" else "cpu"
        for state, mdl in zip(["last", "best"], [clf, clf.load_from_checkpoint(checkpoint.best_model_path)]):
            print(f"------ {state} ------")
            mdl.to(device)
            mdl.eval()
            for version, dl in dl_list:
                y_class_pred, y_list, p_perf_list, z_list = [], [], [], []
                for batch_idx, batch in enumerate(dl):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred_dict = mdl.predict_step(batch=batch, batch_idx=batch_idx)

                    # GT model.
                    y_class_pred.append(pred_dict["p_class"].argmax(dim=-1).cpu())
                    y_list.append(batch["y"].cpu())

                    # AP model.
                    p_perf, z = pred_dict.get("p_perf", None), batch.get("z", None)
                    if p_perf is not None and z is not None:
                        p_perf_list.append(p_perf.cpu())
                        z_list.append(z.cpu())

                y_class_pred, y = torch.concat(y_class_pred).numpy(), torch.concat(y_list).numpy()

                # Compute gt accuracy.
                gt_acc = accuracy_score(y_true=y, y_pred=y_class_pred)
                gt_acc_name = f"gt_{version}_acc_{state}"
                print(f"{gt_acc_name}: {gt_acc}")
                log_metric(gt_acc_name, gt_acc)

                # Compute gt balanced accuracy.
                gt_bal_acc = balanced_accuracy_score(y_true=y, y_pred=y_class_pred)
                gt_bal_acc_name = f"gt_{version}_bal_acc_{state}"
                print(f"{gt_bal_acc_name}: {gt_bal_acc}")
                log_metric(gt_bal_acc_name, gt_bal_acc)

                if len(p_perf_list) > 0 and len(z) > 0:
                    p_perf, z = torch.concat(p_perf_list).numpy(), torch.concat(z_list).numpy()
                    is_labeled = z != -1
                    is_true = y[:, None] == z
                    is_true = is_true.ravel()[is_labeled.ravel()]
                    p_perf = p_perf.ravel()[is_labeled.ravel()]
                    y_ap_pred = p_perf > 0.5

                    # Compute ap accuracy.
                    ap_acc = accuracy_score(y_true=is_true, y_pred=y_ap_pred)
                    ap_acc_name = f"ap_{version}_acc_{state}"
                    print(f"{ap_acc_name}: {ap_acc}")
                    log_metric(ap_acc_name, ap_acc)

                    # Compute ap balanced accuracy.
                    ap_bal_acc = balanced_accuracy_score(y_true=is_true, y_pred=y_ap_pred)
                    ap_bal_acc_name = f"ap_{version}_bal_acc_{state}"
                    print(f"{ap_bal_acc_name}: {ap_bal_acc}")
                    log_metric(ap_bal_acc_name, ap_bal_acc)

                    # Compute ap auroc.
                    ap_auroc = roc_auc_score(y_true=is_true, y_score=p_perf)
                    ap_auroc_name = f"ap_{version}_auroc_{state}"
                    print(f"{ap_auroc_name}: {ap_auroc}")
                    log_metric(ap_auroc_name, ap_auroc)

                    # Compute ap aupr.
                    ap_aupr_0 = average_precision_score(y_true=is_true, y_score=p_perf)
                    ap_aupr_0_name = f"ap_{version}_aupr_0_{state}"
                    print(f"{ap_aupr_0_name}: {ap_aupr_0}")
                    log_metric(ap_aupr_0_name, ap_aupr_0)
                    ap_aupr_1 = average_precision_score(y_true=1 - is_true, y_score=1 - p_perf)
                    ap_aupr_1_name = f"ap_{version}_aupr_1_{state}"
                    print(f"{ap_aupr_1_name}: {ap_aupr_1}")
                    log_metric(ap_aupr_1_name, ap_aupr_1)


if __name__ == "__main__":
    evaluate()
