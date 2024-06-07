import hydra
import os
import sys
import warnings
import torch

from hydra.utils import instantiate, get_class
from lightning.pytorch import seed_everything, Trainer
from torch import set_float32_matmul_precision
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.distributions import Beta, Uniform

sys.path.append("../")
warnings.filterwarnings("ignore")
set_float32_matmul_precision("medium")


@hydra.main(config_path="../conf", config_name="simulate", version_base=None)
def simulate_annotators(cfg):
    from maml.classifiers import AggregateClassifier
    from maml.architectures import maml_net_params

    seed_everything(cfg.seed)

    # Build dataset.
    ds_train = instantiate(cfg.simulation.class_definition, version="train", aggregation_method="ground-truth")
    transform = ds_train.transform if ds_train else "auto"
    ds_valid = instantiate(
        cfg.simulation.class_definition,
        version="valid",
        aggregation_method="ground-truth",
        transform=transform,
    )
    ds_test = instantiate(
        cfg.simulation.class_definition,
        version="test",
        aggregation_method="ground-truth",
        transform=transform,
    )
    ds = ConcatDataset([ds_train, ds_valid, ds_test])
    y = torch.concat([ds_train.y, ds_valid.y, ds_test.y])

    # Set remaining parameters for simulating annotators.
    data_loader_dict = {
        "batch_size": cfg.simulation.batch_size,
        "num_workers": cfg.simulation.max_epochs,
        "shuffle": True,
    }

    # Sample `train_ratios` according to different beta distributions.
    alphas = Uniform(low=cfg.simulation.alpha_1, high=cfg.simulation.beta_1).sample_n(cfg.n_annotators)
    betas = Uniform(low=cfg.simulation.alpha_2, high=cfg.simulation.beta_2).sample_n(cfg.n_annotators)
    train_ratios = torch.empty(cfg.n_annotators, ds_train.get_n_classes())
    for i, (alpha, beta) in enumerate(zip(alphas, betas)):
        train_ratios_dist = Beta(concentration1=alpha, concentration0=beta)
        train_ratios[i] = train_ratios_dist.sample_n(ds_train.get_n_classes())
        while train_ratios[i].sum() * len(ds) < cfg.simulation.min_samples:
            print("Resample")
            train_ratios[i] = train_ratios_dist.sample_n(ds_train.get_n_classes())

    # Sample number of training epochs.
    max_epochs = torch.randint(low=1, high=cfg.simulation.max_epochs+1, size=(cfg.n_annotators,))

    # Sample learning rates.
    lr_max = cfg.simulation.lr_max
    lr_min = cfg.simulation.lr_min
    learning_rates = torch.rand( size=(cfg.n_annotators,)) * (lr_max + lr_min) + lr_min

    data_loader_dict_predict = data_loader_dict.copy()
    data_loader_dict_predict["shuffle"] = False
    dataloader = DataLoader(dataset=ds, **data_loader_dict_predict)
    data_loader_dict_train = data_loader_dict.copy()
    data_loader_dict_train["drop_last"] = True
    z = []
    for idx, (train_ratio, max_epoch, lr) in enumerate(zip(train_ratios, max_epochs, learning_rates)):
        subset_indices = []
        for c, tr in enumerate(train_ratio):
            c_idx = torch.argwhere(y == c)
            c_idx = c_idx[torch.randint(low=0, high=len(c_idx), size=(int(tr * len(c_idx)),))]
            subset_indices.append(c_idx)
        subset_indices = torch.concat(subset_indices).ravel()
        subset = Subset(dataset=ds, indices=subset_indices)
        dataloader_subset = DataLoader(dataset=subset, **data_loader_dict_train)

        # Adjust `max_epochs` according to individual subset.
        trainer_dict_subset = {
            "max_epochs": int(max_epoch),
            "accelerator": cfg.accelerator,
            "logger": False,
            "enable_checkpointing": False
        }
        trainer_subset = Trainer(**trainer_dict_subset)

        # Adjust `lr` according to individual subset.
        params_dict_subset = maml_net_params(
            gt_name=cfg.architecture.name,
            gt_params_dict={"n_classes": ds_train.get_n_classes(), **cfg.architecture.params},
            classifier_name="aggregate",
            optimizer=get_class(cfg.simulation.optimizer.class_definition),
            optimizer_gt_dict=cfg.simulation.optimizer.params.update({"lr": float(lr)}),
        )

        # Create `clf` for individual subset.
        clf_sub = AggregateClassifier(**params_dict_subset)

        # Train `clf` with individual subset.
        print(f"idx={idx}, train_ratio={train_ratio.mean()}, max_epochs={max_epoch}, lr={lr}")
        trainer_subset.fit(model=clf_sub, train_dataloaders=dataloader_subset)

        # Obtain predictions from individual subset.
        trainer_subset.test(model=clf_sub, dataloaders=dataloader)
        y_pred_list = trainer_subset.predict(model=clf_sub, dataloaders=dataloader)
        z.append(torch.concat([y_pred_dict["p_class"].argmax(dim=-1) for y_pred_dict in y_pred_list]))

    # Save obtained annotations.
    z = torch.stack(z, dim=1)
    torch.save(z, os.path.join(cfg.simulation.class_definition.root, f"{cfg.simulation.name}.pt"))


if __name__ == "__main__":
    simulate_annotators()
