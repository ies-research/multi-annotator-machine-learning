'''
This Python script writes the Bash scripts for reproducing the results of the hyperparameter and benchmark study.
Before executing this script.
'''
import os
from itertools import product


def write_commands(
    config_combs: list,
    path_python_file: str = ".",
    directory: str = ".",
    use_slurm: bool = True,
    mem: str = "20gb",
    max_n_parallel_jobs: int = 12,
    cpus_per_task: int = 4,
    slurm_logs_path: str = "slurm_logs",
):
    """
    Writes Bash scripts for the experiments.

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
    use_gpu : bool
        Flag whether to use a GPU. Only used if `use_slurm=True`.
    slurm_logs_path : str
        Path to the directory where the SLURM logs are to saved. Only used if `use_slurm=True`.
    """
    for cfg_dict in config_combs:
        keys, values = zip(*cfg_dict["params"].items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        n_jobs = len(permutations_dicts)
        if max_n_parallel_jobs > n_jobs:
            max_n_parallel_jobs = n_jobs
        job_name = f"{cfg_dict['data']}_{cfg_dict['experiment_name']}"
        filename = os.path.join(directory, f"{job_name}.sh")
        commands = [
            f"#!/usr/bin/env bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --array=1-{n_jobs}%{max_n_parallel_jobs}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --ntasks=1",
            f"#SBATCH --get-user-env",
            f"#SBATCH --time=12:00:00",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --partition=main",
            f"#SBATCH --output={slurm_logs_path}/{job_name}_%A_%a.log",
        ]
        if cfg_dict["accelerator"] == "gpu":
            commands += [
                f"#SBATCH --gres=gpu:1",
                f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{13})) p" {filename})"',
                f"exit 0",
            ]
        else:
            commands += [
                f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{12})) p" {filename})"',
                f"exit 0",
            ]
        python_command = f"srun python"
        if not use_slurm:
            commands = [commands[0]]
            python_command = f"python"
        for param_dict in permutations_dicts:
            commands.append(
                f"{python_command} "
                f"{path_python_file} "
                f"experiment_name={cfg_dict['experiment_name']} "
                f"accelerator={cfg_dict['accelerator']} "
                f"data={cfg_dict['data']} "
                f"architecture={cfg_dict['architecture']} "
                f"ssl_model={cfg_dict['ssl_model']} "
            )
            for k, v in param_dict.items():
                commands[-1] += f"{k}={v} "
            if not use_slurm:
                commands.append("wait")
        print(filename)
        with open(filename, "w") as f:
            for item in commands:
                f.write("%s\n" % item)


if __name__ == "__main__":
    # TODO: Update the default arguments of the `write_commands` function below to fit your machine.
    path_python_file = "your/absolute/path/to/perform_experiment.py"
    directory = "your/absolute/path/to/bash_scripts/"
    use_slurm = True
    mem = "10gb"
    max_n_parallel_jobs = 110
    cpus_per_task = 4
    accelerator = "cpu"
    slurm_logs_path = ""

    # List of seeds to ensure reproducibility.
    seeds = list(range(10))

    # ================================= Create bash scripts for hyperparameter search. ================================
    config_combs = [
        {
            "experiment_name": "hyperparameter_search",
            "data": "dopanim",
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "accelerator": accelerator,
            "params": {
                "seed": seeds,
                "classifier": ["ground_truth"],
                "data.train_batch_size": [32, 64, 128],
                "data.optimizer.gt_params.lr": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
                "data.optimizer.gt_params.weight_decay": [0, 1e-5, 1e-4]
            },
        },
    ]
    write_commands(
        path_python_file=path_python_file,
        directory=directory,
        config_combs=config_combs,
        slurm_logs_path=slurm_logs_path,
        max_n_parallel_jobs=max_n_parallel_jobs,
        mem=mem,
        cpus_per_task=cpus_per_task,
        use_slurm=use_slurm,
    )

    # ================================= Create bash scripts for benchmark. ============================================
    classifiers = [
        "annot_mix",
        "conal",
        "crowd_layer",
        "crowdar",
        "ground_truth",
        "madl",
        "majority_vote",
        "trace_reg",
        "geo_reg_f",
        "geo_reg_w",
        "union_net",
    ]

    for variant in ["worst-1", "worst-2", "worst-var", "rand-1", "rand-2", "rand-var", "full"]:
        config_combs = [
            {
                "experiment_name": f"benchmark_{variant}",
                "data": "dopanim",
                "architecture": "dino_head",
                "ssl_model": "dino_backbone",
                "accelerator": accelerator,
                "params": {
                    "seed": seeds,
                    "classifier": classifiers,
                    "data.class_definition.variant": [variant],
                },
            },
        ]
        write_commands(
            path_python_file=path_python_file,
            directory=directory,
            config_combs=config_combs,
            slurm_logs_path=slurm_logs_path,
            max_n_parallel_jobs=max_n_parallel_jobs,
            mem=mem,
            cpus_per_task=cpus_per_task,
            use_slurm=use_slurm,
        )
    # ================= Create bash scripts for case study on beyond hard class labels. ===============================
    for variant in ["worst-1", "worst-2", "worst-var", "rand-1", "rand-2", "rand-var", "full"]:
        config_combs = [
            {
                "experiment_name": f"beyond_hard_labels_{variant}",
                "data": "dopanim",
                "architecture": "dino_head",
                "ssl_model": "dino_backbone",
                "accelerator": accelerator,
                "params": {
                    "seed": seeds,
                    "classifier": ["majority_vote"],
                    "data.class_definition.variant": [variant],
                    "data.class_definition.annotation_type": ["probabilities"],
                },
            },
        ]
        write_commands(
            path_python_file=path_python_file,
            directory=directory,
            config_combs=config_combs,
            slurm_logs_path=slurm_logs_path,
            max_n_parallel_jobs=max_n_parallel_jobs,
            mem=mem,
            cpus_per_task=cpus_per_task,
            use_slurm=use_slurm,
        )

    # ======================= Create bash scripts for case study on annotator metadata. ===============================
    for variant in ["worst-1", "worst-2", "worst-var", "rand-1", "rand-2", "rand-var", "full"]:
        config_combs = [
            {
                "experiment_name": f"annotator_metadata_{variant}",
                "data": "dopanim",
                "architecture": "dino_head",
                "ssl_model": "dino_backbone",
                "accelerator": accelerator,
                "params": {
                    "seed": seeds,
                    "classifier": ["annot_mix"],
                    "classifier.annotators": ["metadata"],
                    "data.class_definition.variant": [variant],
                },
            },
        ]
        write_commands(
            path_python_file=path_python_file,
            directory=directory,
            config_combs=config_combs,
            slurm_logs_path=slurm_logs_path,
            max_n_parallel_jobs=max_n_parallel_jobs,
            mem=mem,
            cpus_per_task=cpus_per_task,
            use_slurm=use_slurm,
        )