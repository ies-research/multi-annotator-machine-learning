import numpy as np
import os
import pandas as pd

from scipy.stats import loguniform, uniform
from scipy.stats import qmc


def generate_sobol_configs(hyperparam_options, n_configs=5, seed=0):
    """
    Generate hyperparameter configurations from a Sobol sequence.

    Parameters
    ----------
    hyperparam_options : dict
        A dictionary with hyperparameter names as keys and either:
          - lists of possible values, or
          - scipy.stats distributions (with a 'ppf' method)
        as values.

        Example:
            {
                "batch_size": [16, 32, 64],
                "optimizer": ["sgd", "adam"],
                "learning_rate": loguniform(1e-4, 1e-1),
                "dropout_rate": uniform(0.0, 0.5)
            }

    n_configs : int, optional
        Number of configurations to generate. Default is 5.

    seed : int, optional
        Seed for the Sobol sequence scramble. Default is 0.

    Returns
    -------
    list of dict
        List of hyperparameter configurations.
    """
    # Number of hyperparameters
    hyperparam_names = list(hyperparam_options.keys())
    dimension = len(hyperparam_names)

    # Create a Sobol engine
    sobol_engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)

    # Generate n_configs samples in [0, 1]^dimension
    sobol_samples = sobol_engine.random(n_configs)

    configs = []
    for i in range(n_configs):
        config = {}
        for j, hyperparam_name in enumerate(hyperparam_names):
            choices = hyperparam_options[hyperparam_name]
            u = sobol_samples[i, j]  # Sobol value in [0, 1]

            if isinstance(choices, list):
                # Map [0, 1] -> discrete index
                idx = int(u * len(choices))
                # Safeguard against idx == len(choices) if u == 1.0
                if idx == len(choices):
                    idx -= 1
                sampled_value = choices[idx]

            elif hasattr(choices, "ppf"):
                # For continuous distributions, use the inverse-CDF (ppf)
                sampled_value = choices.ppf(u)

            else:
                raise ValueError(
                    f"Unsupported type for hyperparameter '{hyperparam_name}'. "
                    "Expected a list or a distribution with 'ppf' method."
                )

            config[hyperparam_name] = sampled_value

            # Handling dependent hyperparameters
            if hyperparam_name == "data.optimizer.gt_params.lr":
                config["data.optimizer.ap_params.lr"] = sampled_value
            if hyperparam_name == "data.max_epochs":
                config["data.lr_scheduler.params.T_max"] = sampled_value

        configs.append(config)

    return configs


def write_commands(
    config_combs: list,
    path_python_file: str = ".",
    directory: str = ".",
    use_slurm: bool = True,
    mem: str = "20gb",
    max_n_parallel_jobs: int = 12,
    cpus_per_task: int = 4,
    slurm_logs_path: str = "slurm_logs",
    max_param_configs: int = 900,
    partition: str = "main",
):
    """
    Writes Bash scripts for the experiments, splitting the parameter configurations across multiple files
    if the total number exceeds max_param_configs.

    Parameters
    ----------
    config_combs : list
        A list of dictionaries defining the configurations of the experiments.
    path_python_file : str, default="."
        Absolute path to the Python file to be executed.
    directory : str
        Path to the directory where the Bash scripts are to be saved.
    use_slurm : bool
        Flag whether SLURM shall be used.
    mem : str
        RAM size allocated for each experiment. Only used if `use_slurm=True`.
    max_n_parallel_jobs : int
        Maximum number of experiments executed in parallel. Only used if `use_slurm=True`.
    cpus_per_task : int
        Number of CPUs allocated for each experiment. Only used if `use_slurm=True`.
    slurm_logs_path : str
        Path to the directory where the SLURM logs are to be saved. Only used if `use_slurm=True`.
    max_param_configs : int
        Maximum number of parameter configurations to include per file.
        If the total number of parameter configurations exceeds this number, multiple files are created.
    """
    from itertools import product
    import os

    for cfg_dict in config_combs:
        # Extract parameter configurations.
        permutations_dicts = cfg_dict.pop("params")
        if isinstance(permutations_dicts, dict):
            keys, values = zip(*permutations_dicts.items())
            permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        total_configs = len(permutations_dicts)

        # Split the configurations into chunks of size max_param_configs.
        for chunk_index in range(0, total_configs, max_param_configs):
            chunk = permutations_dicts[chunk_index : chunk_index + max_param_configs]
            n_jobs_chunk = len(chunk)

            # Adjust max_n_parallel_jobs for this chunk if needed.
            current_max_parallel_jobs = max_n_parallel_jobs
            if current_max_parallel_jobs > n_jobs_chunk:
                current_max_parallel_jobs = n_jobs_chunk

            # Define the filename. If there are multiple files for the same experiment,
            # append a part number.
            if total_configs > max_param_configs:
                filename = os.path.join(
                    directory, f"{cfg_dict['experiment_name']}_part{chunk_index // max_param_configs + 1}.sh"
                )
            else:
                filename = os.path.join(directory, f"{cfg_dict['experiment_name']}.sh")

            commands = [f"#!/usr/bin/env bash"]
            if use_slurm:
                commands.extend(
                    [
                        f"#SBATCH --job-name={cfg_dict['experiment_name']}",
                        f"#SBATCH --array=1-{n_jobs_chunk}%{current_max_parallel_jobs}",
                        f"#SBATCH --mem={mem}",
                        f"#SBATCH --ntasks=1",
                        f"#SBATCH --get-user-env",
                        f"#SBATCH --time=24:00:00",
                        f"#SBATCH --cpus-per-task={cpus_per_task}",
                        f"#SBATCH --partition={partition}",
                        f"#SBATCH --output={slurm_logs_path}/{cfg_dict['experiment_name']}_%A_%a.log",
                    ]
                )
                if cfg_dict.get("accelerator", None) == "gpu":
                    commands.extend(
                        [
                            f"#SBATCH --gres=gpu:1",
                            f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{13})) p" {filename})"',
                            "exit 0",
                        ]
                    )
                else:
                    commands.extend(
                        [
                            f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{12})) p" {filename})"',
                            "exit 0",
                        ]
                    )
            else:
                # For non-SLURM execution, only include the shebang.
                commands = [commands[0]]
            python_command = f"srun python" if use_slurm else "python"

            # Add a command for each parameter configuration in this chunk.
            for param_idx, param_dict in enumerate(chunk):
                sleep_time = 0 if param_idx == 0 else 20 + 0.1 * param_idx
                cmd_line = f"sleep {sleep_time}; {python_command} {path_python_file} "
                # Merge the common config with the parameter-specific config.
                merged_params = {**cfg_dict, **param_dict}
                for k, v in merged_params.items():
                    cmd_line += f"{k}={v} "
                commands.append(cmd_line)
                if not use_slurm:
                    commands.append("wait")

            print(f"{filename}: {n_jobs_chunk} jobs")
            with open(filename, "w") as f:
                for item in commands:
                    f.write(f"{item}\n")


if __name__ == "__main__":
    # --------------------------- TODO: Update the following variables to fit your setup. -----------------------------

    # There are three experiment types:
    # - default: perform experiments with default data-agnostic HPCs,
    # - default_data: perform experiments with default data-specific HPCs,
    # - hyperparameter_search: perform experiments with HPCs generated by Sobol sequences.
    # You need to execute the script, with each of these experiment types to reproduce all results.
    experiment_type = "hyperparameter_search"

    # Path to the Python file `perform_experiment.py`.
    path_python_file = "/mnt/home/mherde/projects/github/multi-annotator-machine-learning/empirical_evaluation/python_scripts/perform_experiment.py"

    # Path to the directory, where the generated scripts for execution with or without SLURM are to be saved.
    directory = "../crowd_hpo_scripts"

    # Path to the directory, where the datasets have been stored.
    data_sets_path = "/mnt/work/mherde/maml/data"

    # Path to the directory, where the `mlflow` saves the experimental results.
    mlruns_path = "/mnt/work/mherde/maml/crowd_hpo"

    # Path to the directory, where the `hydra` saves the experimental logs.
    hydra_run_path = "/mnt/work/mherde/maml/crowd-opt/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"

    # Path to the directory for the cache of the experimental results. It's used to create only scripts for experiments not yet performed.
    cache_path = "../crowd_hpo_cache"

    # String whether CPUs ("cpu") or GPUs ("gpu") are to be used for experimentation
    accelerator = "cpu"

    # Flag whether SLURM with job arrays is to be used.
    use_slurm = True

    # SLURM parameter for limiting the maximum memory per job.
    mem = "10gb"

    # SLURM parameter for limiting the maximum number of jobs per array.
    max_n_parallel_jobs = 200

    # SLURM parameter for limiting the maximum number of CPUs per job.
    cpus_per_task = 2

    # SLURM parameter specifying the directory where any SLURM logs are stored.
    slurm_logs_path = "/mnt/work/mherde/maml/crowd_opt_logs"

    # SLURM parameter specifying on which partition the experiments are performed.
    partition = "main"

    # ---------------------------------- Define general experimental setup. ----------------------------------
    seed_list = list(range(5))
    classifiers = [
        "ground_truth",
        "majority_vote",
        "dawid_skene",
        "crowd_layer",
        "trace_reg",
        "conal",
        "union_net_a",
        "union_net_b",
        "madl",
        "geo_reg_f",
        "geo_reg_w",
        "crowd_ar",
        "annot_mix",
        "coin_net",
    ]
    data_set_dict = {
        "dopanim_worst-1": {
            "seed": 0,
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "worst-1",
        },
        "dopanim_worst-2": {
            "seed": 1,
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "worst-2",
        },
        "dopanim_worst-var": {
            "seed": 2,
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "worst-var",
        },
        "dopanim_rand-1": {
            "seed": 3,
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "rand-1",
        },
        "dopanim_rand-2": {
            "seed": 4,
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "rand-2",
        },
        "dopanim_rand-var": {
            "seed": 5,
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "rand-var",
        },
        "dopanim_full": {
            "seed": 6,
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "full",
        },
        "music_genres_full": {
            "seed": 7,
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "ssl_model": "none",
            "variant": "full",
        },
        "music_genres_worst-1": {
            "seed": 10,
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "ssl_model": "none",
            "variant": "worst-1",
        },
        "music_genres_worst-2": {
            "seed": 12,
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "ssl_model": "none",
            "variant": "worst-2",
        },
        "music_genres_worst-var": {
            "seed": 13,
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "ssl_model": "none",
            "variant": "worst-var",
        },
        "music_genres_rand-1": {
            "seed": 14,
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "ssl_model": "none",
            "variant": "rand-1",
        },
        "music_genres_rand-2": {
            "seed": 15,
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "ssl_model": "none",
            "variant": "rand-2",
        },
        "music_genres_rand-var": {
            "seed": 16,
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "ssl_model": "none",
            "variant": "rand-var",
        },
        "spc_full": {"seed": 8, "data": "spc", "architecture": "tabnet_spc", "ssl_model": "none", "variant": "full"},
        "spc_worst-1": {
            "seed": 17,
            "data": "spc",
            "architecture": "tabnet_spc",
            "ssl_model": "none",
            "variant": "worst-1",
        },
        "spc_worst-2": {
            "seed": 18,
            "data": "spc",
            "architecture": "tabnet_spc",
            "ssl_model": "none",
            "variant": "worst-2",
        },
        "spc_worst-var": {
            "seed": 19,
            "data": "spc",
            "architecture": "tabnet_spc",
            "ssl_model": "none",
            "variant": "worst-var",
        },
        "spc_rand-1": {
            "seed": 20,
            "data": "spc",
            "architecture": "tabnet_spc",
            "ssl_model": "none",
            "variant": "rand-1",
        },
        "spc_rand-2": {
            "seed": 21,
            "data": "spc",
            "architecture": "tabnet_spc",
            "ssl_model": "none",
            "variant": "rand-2",
        },
        "spc_rand-var": {
            "seed": 22,
            "data": "spc",
            "architecture": "tabnet_spc",
            "ssl_model": "none",
            "variant": "rand-var",
        },
        "label_me_full": {
            "seed": 9,
            "data": "label_me",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "full",
        },
        "label_me_worst-1": {
            "seed": 23,
            "data": "label_me",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "worst-1",
        },
        "label_me_worst-2": {
            "seed": 24,
            "data": "label_me",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "worst-2",
        },
        "label_me_worst-var": {
            "seed": 25,
            "data": "label_me",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "worst-var",
        },
        "label_me_rand-1": {
            "seed": 26,
            "data": "label_me",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "rand-1",
        },
        "label_me_rand-2": {
            "seed": 27,
            "data": "label_me",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "rand-2",
        },
        "label_me_rand-var": {
            "seed": 28,
            "data": "label_me",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "variant": "rand-var",
        },
        "reuters_full": {
            "seed": 11,
            "data": "reuters",
            "architecture": "tabnet_reuters",
            "ssl_model": "none",
            "variant": "full",
        },
        "reuters_worst-1": {
            "seed": 29,
            "data": "reuters",
            "architecture": "tabnet_reuters",
            "ssl_model": "none",
            "variant": "worst-1",
        },
        "reuters_worst-2": {
            "seed": 30,
            "data": "reuters",
            "architecture": "tabnet_reuters",
            "ssl_model": "none",
            "variant": "worst-2",
        },
        "reuters_worst-var": {
            "seed": 31,
            "data": "reuters",
            "architecture": "tabnet_reuters",
            "ssl_model": "none",
            "variant": "worst-var",
        },
        "reuters_rand-1": {
            "seed": 32,
            "data": "reuters",
            "architecture": "tabnet_reuters",
            "ssl_model": "none",
            "variant": "rand-1",
        },
        "reuters_rand-2": {
            "seed": 33,
            "data": "reuters",
            "architecture": "tabnet_reuters",
            "ssl_model": "none",
            "variant": "rand-2",
        },
        "reuters_rand-var": {
            "seed": 34,
            "data": "reuters",
            "architecture": "tabnet_reuters",
            "ssl_model": "none",
            "variant": "rand-var",
        },
        # "cifar_100_n": {"seed": 35, "data": "cifar_100_n", "architecture": "dino_head", "ssl_model": "dino_backbone", "variant": None},
        "image_net_15_n": {
            "seed": 36,
            "data": "image_net_15_n",
            "architecture": "tabnet_image_net_15_n",
            "ssl_model": "none",
            "variant": None,
        },
    }

    # ----------------------------------- Define search spaces for hyperparameters. -----------------------------------
    if experiment_type == "hyperparameter_search":

        for key in data_set_dict:
            data_set_dict[key]["hp_ranges"] = {
                "data.optimizer.class_definition": ["torch.optim.RAdam"],
                "data.max_epochs": [5, 30, 50],
                "data.train_batch_size": [16, 32, 64],
                "data.eval_batch_size": [128],
                "data.lr_scheduler.class_definition": ["torch.optim.lr_scheduler.CosineAnnealingLR"],
                "data.optimizer.gt_params.lr": loguniform(1e-4, 1e-1),
                "data.optimizer.gt_params.weight_decay": loguniform(1e-6, 1e-3),
                "data.optimizer.ap_params.weight_decay": [0],
                "architecture.params.dropout_rate": uniform(0, 0.5),
            }
    elif experiment_type == "default_data":

        data_set_dict["dopanim_worst-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0012025892642767],
            "data.optimizer.gt_params.weight_decay": [8.427803994702244e-06],
            "data.optimizer.ap_params.lr": [0.0012025892642767],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.4146318528801203],
        }
        data_set_dict["dopanim_worst-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0724927599359765],
            "data.optimizer.gt_params.weight_decay": [8.556786072005315e-06],
            "data.optimizer.ap_params.lr": [0.0724927599359765],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.3396366885863244],
        }
        data_set_dict["dopanim_worst-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0005677113487551],
            "data.optimizer.gt_params.weight_decay": [1.087323211440524e-06],
            "data.optimizer.ap_params.lr": [0.0005677113487551],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.4241075105965137],
        }
        data_set_dict["dopanim_rand-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0478575970005997],
            "data.optimizer.gt_params.weight_decay": [2.028481853053322e-05],
            "data.optimizer.ap_params.lr": [0.0478575970005997],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.3733238480053842],
        }
        data_set_dict["dopanim_rand-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0240097136357499],
            "data.optimizer.gt_params.weight_decay": [3.58474704747276e-05],
            "data.optimizer.ap_params.lr": [0.0240097136357499],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [64],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.4928859313949942],
        }
        data_set_dict["dopanim_rand-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0016981528229203],
            "data.optimizer.gt_params.weight_decay": [5.099066983264771e-06],
            "data.optimizer.ap_params.lr": [0.0016981528229203],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.3383297156542539],
        }
        data_set_dict["dopanim_full"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0047551872706569],
            "data.optimizer.gt_params.weight_decay": [0.0009767409102608],
            "data.optimizer.ap_params.lr": [0.0047551872706569],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.3637089189141989],
        }

        data_set_dict["music_genres_worst-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0097314556350408],
            "data.optimizer.gt_params.weight_decay": [0.0007748511068316],
            "data.optimizer.ap_params.lr": [0.0097314556350408],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [64],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.1845566583797335],
        }
        data_set_dict["music_genres_worst-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0198885065261295],
            "data.optimizer.gt_params.weight_decay": [9.39923573686536e-06],
            "data.optimizer.ap_params.lr": [0.0198885065261295],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.346677865833044],
        }
        data_set_dict["music_genres_worst-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0331919106915925],
            "data.optimizer.gt_params.weight_decay": [4.343304960250691e-06],
            "data.optimizer.ap_params.lr": [0.0331919106915925],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.3290240643545985],
        }
        data_set_dict["music_genres_rand-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0963480499007724],
            "data.optimizer.gt_params.weight_decay": [5.676821336623866e-05],
            "data.optimizer.ap_params.lr": [0.0963480499007724],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.3753234879113734],
        }
        data_set_dict["music_genres_rand-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0108057719578136],
            "data.optimizer.gt_params.weight_decay": [5.22010436379369e-06],
            "data.optimizer.ap_params.lr": [0.0108057719578136],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.3787592989392578],
        }
        data_set_dict["music_genres_rand-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0921500064948056],
            "data.optimizer.gt_params.weight_decay": [3.082761588991692e-05],
            "data.optimizer.ap_params.lr": [0.0921500064948056],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.3648859527893364],
        }
        data_set_dict["music_genres_full"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0030641520526576],
            "data.optimizer.gt_params.weight_decay": [6.384937740044134e-06],
            "data.optimizer.ap_params.lr": [0.0030641520526576],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.2321276850998401],
        }

        data_set_dict["label_me_worst-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0336677677922686],
            "data.optimizer.gt_params.weight_decay": [1.3028617086928858e-05],
            "data.optimizer.ap_params.lr": [0.0336677677922686],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.4760055867955088],
        }
        data_set_dict["label_me_worst-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0334867952948977],
            "data.optimizer.gt_params.weight_decay": [0.00012417447787],
            "data.optimizer.ap_params.lr": [0.0334867952948977],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.4468165710568428],
        }
        data_set_dict["label_me_worst-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0050648471739756],
            "data.optimizer.gt_params.weight_decay": [9.258901424699031e-06],
            "data.optimizer.ap_params.lr": [0.0050648471739756],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [64],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.480854591820389],
        }
        data_set_dict["label_me_rand-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0357748135002659],
            "data.optimizer.gt_params.weight_decay": [1.246020913824136e-06],
            "data.optimizer.ap_params.lr": [0.0357748135002659],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.1028404184617102],
        }
        data_set_dict["label_me_rand-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0347737401274489],
            "data.optimizer.gt_params.weight_decay": [0.000388243429044],
            "data.optimizer.ap_params.lr": [0.0347737401274489],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [64],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.2943961047567427],
        }
        data_set_dict["label_me_rand-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0253028701374476],
            "data.optimizer.gt_params.weight_decay": [7.374152076702592e-06],
            "data.optimizer.ap_params.lr": [0.0253028701374476],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.3739382452331483],
        }
        data_set_dict["label_me_full"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.014422010112895],
            "data.optimizer.gt_params.weight_decay": [1.0150560624782804e-06],
            "data.optimizer.ap_params.lr": [0.014422010112895],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.2575003001838922],
        }

        data_set_dict["reuters_worst-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0113489173899403],
            "data.optimizer.gt_params.weight_decay": [0.0003279304671303],
            "data.optimizer.ap_params.lr": [0.0113489173899403],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.2274174150079488],
        }
        data_set_dict["reuters_worst-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.019933968007076],
            "data.optimizer.gt_params.weight_decay": [2.064850739599643e-05],
            "data.optimizer.ap_params.lr": [0.019933968007076],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.483272208366543],
        }
        data_set_dict["reuters_worst-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0416402795492026],
            "data.optimizer.gt_params.weight_decay": [0.0004303868492572],
            "data.optimizer.ap_params.lr": [0.0416402795492026],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.1608938793651759],
        }
        data_set_dict["reuters_rand-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0878773631764731],
            "data.optimizer.gt_params.weight_decay": [1.790373167757411e-05],
            "data.optimizer.ap_params.lr": [0.0878773631764731],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.2965719704516232],
        }
        data_set_dict["reuters_rand-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.017158518775817],
            "data.optimizer.gt_params.weight_decay": [6.621704084743511e-05],
            "data.optimizer.ap_params.lr": [0.017158518775817],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.4499105946160853],
        }
        data_set_dict["reuters_rand-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0190212776650518],
            "data.optimizer.gt_params.weight_decay": [2.1949852754172695e-05],
            "data.optimizer.ap_params.lr": [0.0190212776650518],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [50],
            "data.lr_scheduler.params.T_max": [50],
            "architecture.params.dropout_rate": [0.4450469450093806],
        }
        data_set_dict["reuters_full"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0106369310449254],
            "data.optimizer.gt_params.weight_decay": [0.0009305893765619],
            "data.optimizer.ap_params.lr": [0.0106369310449254],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [30],
            "data.lr_scheduler.params.T_max": [30],
            "architecture.params.dropout_rate": [0.357334631960839],
        }

        data_set_dict["spc_worst-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0013308772273279],
            "data.optimizer.gt_params.weight_decay": [2.105923791187603e-06],
            "data.optimizer.ap_params.lr": [0.0013308772273279],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.311977481469512],
        }
        data_set_dict["spc_worst-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.000227344855337],
            "data.optimizer.gt_params.weight_decay": [1.8153700660461776e-05],
            "data.optimizer.ap_params.lr": [0.000227344855337],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.0964406430721283],
        }
        data_set_dict["spc_worst-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0007959915613787],
            "data.optimizer.gt_params.weight_decay": [0.0001875759249311],
            "data.optimizer.ap_params.lr": [0.0007959915613787],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.0138992154970765],
        }
        data_set_dict["spc_rand-1"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0010268209685194],
            "data.optimizer.gt_params.weight_decay": [7.205031526733538e-06],
            "data.optimizer.ap_params.lr": [0.0010268209685194],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [32],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.4308224227279424],
        }
        data_set_dict["spc_rand-2"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0015523323305629],
            "data.optimizer.gt_params.weight_decay": [4.540730088782905e-05],
            "data.optimizer.ap_params.lr": [0.0015523323305629],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [64],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.2670178678818047],
        }
        data_set_dict["spc_rand-var"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0002070586449429],
            "data.optimizer.gt_params.weight_decay": [0.00053680188371],
            "data.optimizer.ap_params.lr": [0.0002070586449429],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [16],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.2529110959731042],
        }
        data_set_dict["spc_full"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0042799206062327],
            "data.optimizer.gt_params.weight_decay": [1.3819760054262464e-05],
            "data.optimizer.ap_params.lr": [0.0042799206062327],
            "data.optimizer.ap_params.weight_decay": [0.0],
            "data.train_batch_size": [64],
            "data.max_epochs": [5],
            "data.lr_scheduler.params.T_max": [5],
            "architecture.params.dropout_rate": [0.4173331866040826],
        }

        data_set_dict["image_net_15_n"]["hp_ranges"] = {
            "data.optimizer.gt_params.lr": [0.0114558483887332],
            "data.optimizer.gt_params.weight_decay": [4.757782538568605e-06],
            "data.optimizer.ap_params.lr": [0.0114558483887332],
            "data.optimizer.ap_params.weight_decay": [0],
            "data.train_batch_size": [64],
            "architecture.params.dropout_rate": [0.4353005206212401],
        }

    elif experiment_type == "default":
        for key in data_set_dict:
            data_set_dict[key]["hp_ranges"] = {}

    # ------------------------------ Define function to generate parameter configurations. ----------------------------
    def get_params(hpc_seed, hp_ranges):
        params = []
        for seed in seed_list:
            for clf in classifiers:
                hpc_addendum = {
                    "seed": [seed],
                    "classifier": [clf],
                    "data.class_definition.realistic_split": [f"cv-5-{seed}"],
                }
                hp_ranges_clf = {}
                if experiment_type == "hyperparameter_search":
                    if clf == "trace_reg":
                        hp_ranges_clf["classifier.params.lmbda"] = loguniform(1e-3, 1e-1)
                    elif clf == "conal":
                        hp_ranges_clf["classifier.params.lmbda"] = loguniform(1e-6, 1e-3)
                        hp_ranges_clf["classifier.embed_size"] = [20, 40, 60, 80]
                    elif clf in ["union_net_a", "union_net_b"]:
                        hp_ranges_clf["classifier.params.epsilon"] = loguniform(1e-6, 1e-4)
                    elif clf == "madl":
                        hp_ranges_clf["classifier.params.eta"] = uniform(0.75, 0.2)
                        hp_ranges_clf["classifier.params.alpha"] = uniform(1.0, 0.5)
                        hp_ranges_clf["classifier.params.beta"] = uniform(0.25, 0.25)
                        hp_ranges_clf["classifier.embed_size"] = [8, 16, 32]
                    elif clf in ["geo_reg_f", "geo_reg_w"]:
                        hp_ranges_clf["classifier.params.lmbda"] = loguniform(1e-4, 1e-2)
                    elif clf == "crowd_ar":
                        hp_ranges_clf["classifier.params.lmbda"] = uniform(0.5, 0.5)
                    elif clf == "annot_mix":
                        hp_ranges_clf["classifier.params.eta"] = uniform(0.75, 0.2)
                        hp_ranges_clf["classifier.params.alpha"] = uniform(0, 2.0)
                    elif clf == "coin_net":
                        hp_ranges_clf["classifier.params.lmbda"] = loguniform(1e-3, 1e-1)
                        hp_ranges_clf["classifier.params.mu"] = loguniform(1e-3, 1e-1)
                        hp_ranges_clf["classifier.params.mu"] = uniform(0.0, 1.0)
                    hpc = hp_ranges | hpc_addendum | hp_ranges_clf
                    n_configs = 50
                else:
                    hpc = hp_ranges | hpc_addendum
                    n_configs = 1
                params += generate_sobol_configs(hyperparam_options=hpc, n_configs=n_configs, seed=hpc_seed)
        return params

    # Datasets: variants of dopanim
    for ds, ds_dict in data_set_dict.items():
        exp_name = f"{experiment_type}_{ds}"
        print(f"############### {exp_name} ###############")
        ds_mlruns_dir = os.path.join(mlruns_path, exp_name)
        os.makedirs(ds_mlruns_dir, exist_ok=True)

        # Sample parameters and filter according existing results.
        params = get_params(ds_dict["seed"], ds_dict["hp_ranges"])
        if cache_path:
            cache_file = os.path.join(cache_path, f"{exp_name}.csv")
            if os.path.isfile(cache_file):
                runs = pd.read_csv(cache_file)

                if len(runs) == len(params):
                    continue

                filtred_params = []
                for d in params:
                    # Only consider keys that are also columns in df_large.
                    keys_to_check = [k for k in d if f"params.{k}" in runs.columns]

                    # If there are no keys to check, we assume the dictionary can't be represented in df_large.
                    if not keys_to_check:
                        filtred_params.append(d)
                        continue

                    # Build a boolean mask where we check each common key.
                    mask = pd.Series([True] * len(runs))
                    for key in keys_to_check:
                        # If the column's type is float, use np.isclose for approximate comparison.
                        if pd.api.types.is_float_dtype(runs[f"params.{key}"]):
                            mask &= np.isclose(runs[f"params.{key}"], d[key])
                        else:
                            mask &= runs[f"params.{key}"].astype(str) == str(d[key])

                    classifier_name = str(d["classifier"])
                    if classifier_name not in ["ground_truth", "majority_vote", "dawid_skene"]:
                        mask &= runs["params.classifier.name"].astype(str) == classifier_name
                    else:
                        aggregation = classifier_name.replace("_", "-")
                        mask &= runs["params.classifier.aggregation_method"].astype(str) == aggregation

                    # If no row in df_large matches all key-value pairs from d, keep d.
                    if not mask.any():
                        filtred_params.append(d)
                params = filtred_params
        if len(params) == 0:
            continue

        # Create configurations.
        config_combs = {
            "mlruns_path": ds_mlruns_dir,
            "hydra.run.dir": hydra_run_path,
            "experiment_name": exp_name,
            "data": ds_dict["data"],
            "data.class_definition.root": data_sets_path,
            "architecture": ds_dict["architecture"],
            "ssl_model": ds_dict["ssl_model"],
            "accelerator": accelerator,
            "params": params,
        }
        if ds_dict["variant"] is not None:
            config_combs["data.class_definition.variant"] = ds_dict["variant"]

        write_commands(
            path_python_file=path_python_file,
            directory=directory,
            config_combs=[config_combs],
            slurm_logs_path=slurm_logs_path,
            max_n_parallel_jobs=max_n_parallel_jobs,
            mem=mem,
            cpus_per_task=cpus_per_task,
            use_slurm=use_slurm,
            partition=partition,
        )
