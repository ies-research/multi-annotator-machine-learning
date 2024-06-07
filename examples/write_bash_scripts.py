'''
This Python script writes the Bash scripts for reproducing the results of the ablation and benchmark study. Before
executing this script, update the default arguments of the `write_commands` function below to fit your machine.
'''
import os
from itertools import product


def write_commands(
    config_combs,
    directory=".",
    use_slurm=True,
    mem="40gb",
    max_n_parallel_jobs=12,
    cpus_per_task=4,
    use_gpu=True,
    slurm_logs_path=".",
):
    """
    Writes Bash scripts for the experiments.

    Parameters
    ----------
    config_combs : list
        A list of dictionaries defining the configurations of the experiments.
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
    python_filename = os.path.join(os.getcwd(), "perform_experiment.py")
    for cfg_dict in config_combs:
        keys, values = zip(*cfg_dict["params"].items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        n_jobs = len(permutations_dicts)
        if max_n_parallel_jobs > n_jobs:
            max_n_parallel_jobs = n_jobs
        job_name = f"{cfg_dict['data']}_{cfg_dict['architecture']}"
        filename = os.path.join(os.getcwd(), directory, f"{job_name}.sh")
        commands = [
            f"#!/usr/bin/env bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --array=1-{n_jobs}%{max_n_parallel_jobs}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --ntasks=1",
            f"#SBATCH --get-user-env",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --partition=main",
            f"#SBATCH --output={slurm_logs_path}/{job_name}_%A_%a.log",
        ]
        if use_gpu:
            commands.append(f"#SBATCH --gres=gpu:1")
        n_lines = len(commands) + 2
        commands.extend([
            f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{n_lines})) p" {filename})"',
            f"exit 0",
        ])
        python_command = f"srun python"
        if not use_slurm:
            commands = [commands[0]]
            python_command = f"python"
        for param_dict in permutations_dicts:
            commands.append(
                f"{python_command} "
                f"{python_filename} "
                f"experiment_name={cfg_dict['experiment_name']} "
                f"data={cfg_dict['data']} "
                f"architecture={cfg_dict['architecture']} "
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
    seeds = list(range(10))

    # ==================================Create bash scripts for benchmark study. ======================================
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
    config_combs_benchmark = [
        {
            "experiment_name": "label_me_benchmark",
            "data": "label_me",
            "architecture": "dino",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "music_genres_benchmark",
            "data": "music_genres",
            "architecture": "tabnet_music_genres",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "cifar_10_h_benchmark",
            "data": "cifar_10_h",
            "architecture": "resnet_18_32x32",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "cifar_10_n_benchmark",
            "data": "cifar_10_n",
            "architecture": "resnet_18_32x32",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "cifar_100_n_benchmark",
            "data": "cifar_100_n",
            "architecture": "resnet_18_32x32",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "letter_sim_benchmark",
            "data": "letter_sim",
            "architecture": "tabnet_letter_sim",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "aloi_sim_benchmark",
            "data": "aloi_sim",
            "architecture": "tabnet_aloi_sim",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "flowers_102_sim_benchmark",
            "data": "flowers_102_sim",
            "architecture": "dino",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "dtd_sim_benchmark",
            "data": "dtd_sim",
            "architecture": "dino",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "trec_6_sim_benchmark",
            "data": "trec_6_sim",
            "architecture": "bert",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
        {
            "experiment_name": "ag_news_sim_benchmark",
            "data": "ag_news_sim",
            "architecture": "bert",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
            },
        },
    ]
    write_commands(
        config_combs=config_combs_benchmark,
    )

    # ==================================Create bash scripts for ablation study. =======================================
    classifiers = ["annot_mix"]
    alpha_values = [0.5, 1.0, 2.0, 4.0]
    config_combs_ablation = [
        {
            "experiment_name": "cifar_10_h_ablation",
            "data": "cifar_10_h",
            "architecture": "resnet_18_32x32",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
                "classifier.params.alpha": alpha_values,
            },
        },
        {
            "experiment_name": "letter_sim_ablation",
            "data": "letter_sim",
            "architecture": "tabnet_letter_sim",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
                "classifier.params.alpha": alpha_values,
            },
        },
        {
            "experiment_name": "dtd_sim_ablation",
            "data": "dtd_sim",
            "architecture": "dino",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
                "classifier.params.alpha": alpha_values,
            },
        },
        {
            "experiment_name": "ag_news_sim_ablation",
            "data": "ag_news",
            "architecture": "bert",
            "params": {
                "seed": seeds,
                "classifier": classifiers,
                "classifier.params.alpha": alpha_values,
            },
        },
    ]
    write_commands(
        config_combs=config_combs_ablation,
    )
