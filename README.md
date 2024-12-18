<div align="center">
  <img src="./docs/images/maml-logo.png" alt="logo" width="200">
</div>

# `maml`: Multi-annotator Machine Learning :busts_in_silhouette:
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch_Lightning-792ee5?logo=pytorch-lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ies-research/multi-annotator-machine-learning/tree/annot-mix"><img alt="dopanim @ NeurIPS 2024" src="https://img.shields.io/badge/GitHub-annot--mix @ ECAI 2024-aqua"></a>
<a href="https://github.com/ies-research/multi-annotator-machine-learning/tree/dopanim"><img alt="dopanim @ NeurIPS 2024" src="https://img.shields.io/badge/GitHub-dopanim @ NeurIPS 2024-aqua"></a>
> Author: Marek Herde

This project implements an ecosystem for multi-annotator learning approaches, aiming to learn from data with
noisy annotations provided by multiple error-prone (mostly human) annotators. This research area is also often 
referred to as learning from crowds or learning from crowd-sourced labels. A graphical model describing this problem
setting is given below.

<div align="left">
  <img src="./docs/images/maml-graphical-model.png" alt="logo" width="1000">
</div>

## Papers Employing `maml` :page_with_curl:	
> - Marek Herde, Lukas Lührs, Denis Huseljic, and Bernhard Sick. Annot-Mix: Learning with Noisy Class Labels from Multiple Annotators via a Mixup Extension. In ECAI, 2024. <a href="https://openreview.net/forum?id=XOGosbxLrz](https://ebooks.iospress.nl/doi/10.3233/FAIA240829"><img alt="dopanim @ NeurIPS 2024" src="https://img.shields.io/badge/Paper-annot--mix @ ECAI 2024-purple"></a> <a href="https://github.com/ies-research/multi-annotator-machine-learning/tree/annot-mix"><img alt="dopanim @ NeurIPS 2024" src="https://img.shields.io/badge/GitHub-annot--mix @ ECAI 2024-aqua"></a>
> - Marek Herde, Denis Huseljic, Lukas Rauch, and Bernhard Sick. dopanim: A Dataset of Doppelganger Animals with Noisy Annotations from Multiple Humans. In NeurIPS, 2024. <a href="https://openreview.net/forum?id=XOGosbxLrz"><img alt="dopanim @ NeurIPS 2024" src="https://img.shields.io/badge/Paper-dopanim @ NeurIPS 2024-purple"></a> <a href="https://github.com/ies-research/multi-annotator-machine-learning/tree/dopanim"><img alt="dopanim @ NeurIPS 2024" src="https://img.shields.io/badge/GitHub-dopanim @ NeurIPS 2024-aqua"></a>


## Multi-annotator Learning Approaches :robot:

Multi-annotator learning approaches estimate annotators' performances for improving neural networks' generalization performances during training:

$$P(y_n, z_{nm} | x_n, a_m) = P(y_n | x_n) \times P(z_{nm} | x_n, a_m, y_n),$$

where

- $P(y_n | x_n)$ represents the class probabilities.
- $Pr(z_{nm} | x_n, a_m, y_n)$ represents the confusion matrix.

The approaches differ in their training and architectures to estimate these two quantities or proxies of them.


| **Approach**    | **Authors**                                                                                             | **Venue (Year)**  | **Annotator Performance Model**                      | **Training**                          |
|-------------|------------------------------------------------------------------------------------------------------|---------------|--------------------------------------------------|-----------------------------------|
| `dawid-skene` | [Dawid and Skene](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2346806)                      | J. R. Stat. (1979)   | confusion matrix per annotator            | em-algorithm + standard cross-entropy |
| `cl`          | [Rodrigues et al.](https://aaai.org/papers/11506-deep-learning-from-crowds/)                        | AAAI (2018)   | noise adaption layer per annotator               | cross-entropy                     |
| `trace-reg`   | [Tanno et al.](https://openaccess.thecvf.com/content_CVPR_2019/html/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.html) | CVPR (2019)   | confusion matrix per annotator                   | cross-entropy + regularization     |
| `conal`       | [Chu et al.](https://ojs.aaai.org/index.php/AAAI/article/view/16730)                                | AAAI (2021)   | confusion matrix per and across annotators       | cross-entropy + regularization     |
| `union-net`   | [Wei et al.](https://ieeexplore.ieee.org/document/9765651/)                                         | TNNLS (2022)  | noise adaption layer across annotators           | cross-entropy                     |
| `geo-reg-w`   | [Ibrahim et al.](https://openreview.net/forum?id=_qVhsWyWB9)                                        | ICLR (2023)   | confusion matrix per annotator                   | cross-entropy + regularization     |
| `geo-reg-f`   | [Ibrahim et al.](https://openreview.net/forum?id=_qVhsWyWB9)                                        | ICLR (2023)   | confusion matrix per annotator                   | cross-entropy + regularization     |
| `madl`        | [Herde et al.](https://openreview.net/forum?id=MgdoxzImlK)                                          | TMLR (2023)   | confusion matrix per instance-annotator pair     | cross-entropy + regularization     |
| `crowd-ar`    | [Cao et al.](https://dl.acm.org/doi/10.1145/3539618.3592007)                                        | SIGIR (2023)  | reliability scalar per instance-annotator pair   | two-model cross-entropy            |
| `annot-mix`   | [Herde et al.](https://ebooks.iospress.nl/doi/10.3233/FAIA240829)                                   | ECAI (2024)   | confusion matrix per instance-annotator pair     | cross-entropy + mixup extension    |


## Datasets :floppy_disk:	

Beyond implementing multi-annotator machine learning approaches, `maml` provides code to download and train with publicly available datasets annotated by multiple error-prone humans (e.g., crowdworkers). The table below lists the currently supported datasets, including their main characteristics. Some of these datasets also come with certain variants to emulate lower or higher levels of annotation noise, e.g., `cifar10n`, `cifar100n`, and `dopanim`.

| **Dataset**         | `spc`                     | `mgc`                    | `labelme`       | `cifar10h`      | `cifar10n`      | `cifar100n`    | `dopanim`     |
|-----------------|-------------------------|------------------------|---------------|---------------|---------------|--------------|-------------|
| Authors  | [Rodrigues<br> et al.](https://www.sciencedirect.com/science/article/abs/pii/S016786551300202X) | [Rodrigues<br> et al.](https://www.sciencedirect.com/science/article/abs/pii/S016786551300202X) | [Rodrigues<br> et al.](https://aaai.org/papers/11506-deep-learning-from-crowds/)   | [Peterson<br> et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Peterson_Human_Uncertainty_Makes_Classification_More_Robust_ICCV_2019_paper.html) | [Wei<br> et al.](https://openreview.net/forum?id=TBWA6PLJZQm) | [Wei<br> et al.](https://openreview.net/forum?id=TBWA6PLJZQm) | [Herde<br> et al.](https://openreview.net/forum?id=XOGosbxLrz)        |
| Venue | PRL<br> (2013) | PRL<br> (2013) | AAAI<br> (2018) | CVPR<br> (2019) | ICLR<br> (2022) | ICLR<br> (2022) | NeurIPS<br> (2024) |
| Data Modality   | text                    | sound                  | image         | image         | image         | image        | image       |
| Training Instances [#] | 4,999           | 700                    | 1,000         | 10,000        | 50,000        | 50,000       | 10,484      |
| Validation Instances [#] | :x:           | :x:                     | 500           | :x:            | :x:            | :x:           | 750         |
| Test Instances [#] | 5,428               | 300                    | 1,188         | 50,000        | 10,000        | 10,000       | 4,500       |
| Classes [#]     | 2                       | 10                     | 8             | 10            | 10            | 100          | 15          |
| Annotators [#]  | 203                     | 42                     | 59            | 2,571         | 747           | 519          | 20          |
| Annotation Platform | AMT                | AMT                    | AMT           | AMT           | AMT           | AMT          | LabelStudio |
| Annotator Metadata | :x:                 | :x:                     | :x:            | :x:            | :x:            | :x:           | :white_check_mark:         |
| Annotation Times | :x:                     | :x:                     | :x:            | :white_check_mark:           | :white_check_mark:           | :white_check_mark:          | :white_check_mark:         |
| Soft Class Labels | :x:                    | :x:                     | :x:            | :x:            | :x:            | :x:           | :white_check_mark:         |
| Annotations per Instance [#] | 5.6        | 4.2                   | 2.5           | 51.4          | 3.0           | 1.0          | 5.0         |
| Annotations per Annotator [#] | 137       | 67                    | 43            | 200           | 201           | 96           | 2,602       |
| Overall Accuracy [%] | 78.9              | 56.0                  | 74.0          | 94.9          | 82.3          | 59.8         | 67.3        |
| Accuracy per Annotator [%] | 77.1         | 73.3                  | 69.2          | 94.9          | 82.1          | 55.6         | 65.6        |

### Code Snippet :computer:
The following code snippet exemplarily demonstrates how to train and test the multi-an:x:tator classifier crowd layer (`cl`) on the dataset music genres classification (`mgc`) via `maml`.
```python
from lightning.pytorch import Trainer
from maml.data import MusicGenres
from torch import nn
from torch.utils.data import DataLoader

# Download the training, validation, and test dataset.
ds_train = MusicGenres(root=".", version="train", download=True)
ds_train = MusicGenres(root=".", version="test", download=True)

# Build data loaders.
dl_train = DataLoader(dataset=ds_train, batch_size=64, shuffle=True, drop_last=True)
dl_test = DataLoader(dataset=ds_train, batch_size=128)

# Define neural network architecture.
gt_embed_x = nn.Sequential(
      nn.Linear(in_features=124, out_features=128),
      nn.BatchNorm1d(num_features=neuron_list[i + 1]),
      nn.ReLU(),
)
gt_output = nn.Linear(in_features=128, out_features=ds_train.get_n_classes())

# Define optimizer with parameters for the ground truth (GT) model estimating class
# probabilities and the annotator performance (AP) model estimating annotators'
# performances.
optimizer = RAdam
optimizer_gt_dict = {"lr": 0.01}
optimizer_ap_dict = {"lr": 0.01}

# Build multi-annotator learning classifier `cl`.
clf = CrowdLayerClassifier(
        n_classes=ds_train.get_n_classes(),
        n_annotators= ds_train.get_n_annotators(),
        gt_embed_x=gt_embed_x,
        gt_output=gt_output,
        optimizer=optimizer,
        optimizer_gt_dict=optimizer_gt_dict,
        optimizer_ap_dict=optimizer_ap_dict,
)

# Train multi-annotator classifier.
trainer = Trainer(max_epochs=50, accelerator="gpu")
trainer.fit(model=clf, train_dataloaders=dl_train)

# Evaluate multi-annotator classifier after the last epoch.
trainer.test(dataloaders=dl_test)
```

## Setup of Conda Environment :snake:
As a prerequisite, we assume to have a Linux distribution as operating system. 

1. Download a [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) version to be installed on your machine. 
2. Setup the environment via
```bash
projectpath$ conda env create -f environment.yml
```
3. Activate the new environment
```bash
projectpath$ conda activate maml
```
4. Verify that the `maml` (multi-annotator machine learning) environment was installed correctly:
```bash
projectpath$ conda env list
```

## Empirical Evaluation :bar_chart:

We provide scripts and Jupyter notebooks to benchmark and visualize multi-annotator machine learning approaches
on datasets annotated by multiple error-prone annotators. 

### Experiments 
The Python script for executing a single experiment is 
[`perform_experiment.py`](empirical_evaluation/python_scripts/perform_experiment.py) and the corresponding main config file 
is [`evaluation`](empirical_evaluation/hydra_configs/evaluation.yaml). In this config file, you also need to specify the `mlruns_path` 
defining the path, where the results will be saved via [`mlflow`](https://mlflow.org/). Further, you can select the 'gpu' or 'cpu' as `accelerator`.
1. Before starting a single experiment or Jupyter notebook, check whether the dataset is downloaded. 
For example, if you want to ensure that the dataset `dopanim` is downloaded, update the `download` flag in its config 
file [`dopanim.yaml`](empirical_evaluation/hydra_configs/data/dopanim.yaml).
2. An experiment can then be started by executing the following commands
```bash
projectpath$ conda activate maml
projectpath$ cd empirical_evaluation/python_scripts
projectpath/empirical_evaluation/python_scripts$ python perform_experiment.py data=dopanim data.class_definition.variant="full" classifier=majority_vote seed=0
````
3. Since there are many possible experimental configurations, including repetitions with different seeds, you can
create Bash scripts. As an example, you can reproduce the experiments of the `dopanim` paper by following the instructions
in [`write_bash_scripts.py`](empirical_evaluation/python_scripts/write_bash_scripts.py) and then execute the following commands
```bash
projectpath$ conda activate maml
projectpath$ cd empirical_evaluation/python_scripts
projectpath/empirical_evaluation/python_scripts$ python write_bash_scripts.py
```
4. There is a bash script for the hyperparameter search, each dataset variant of the benchmark, and use cases. For 
example, executing the benchmark experiments for the variant `full` via SLURM can be done according to
```bash
projectpath$ conda activate maml
projectpath$ sbatch path_to_bash_scripts/dopanim_benchmark_full.sh
```

### Results
Once an experiment is completed, its associated results can be loaded via [`mlflow`](https://mlflow.org/). 
To get a tabular presentation of these results, you need to start the Jupyter notebook 
[`tabular_results.ipynb`](examples/tabular_results.ipynb) and follow its instructions.
```bash
projectpath$ conda activate maml
projectpath$ cd empirical_evaluation/jupyter_notebooks
projectpath/empirical_evaluation/jupyter_notebooks$ jupyter-notebook tabular_results.ipynb
```

## Structure :classical_building:
- [`empirical_evaluation`](empirical_evaluation): scripts to reproduce or adjust our empirical evaluation, including 
  the benchmark and case studies
  - [`hydra_configs`](empirical_evaluation/hydra_configs): collection of [`hydra`](https://hydra.cc/docs/intro/) config
    files for defining hyperparameters
    - [`architecture`](empirical_evaluation/hydra_configs/architecture): config group of config files for network 
      architectures
    - [`classifier`](empirical_evaluation/hydra_configs/classifier): config group of config files for multi-annotator 
      classification approaches
    - [`data`](empirical_evaluation/hydra_configs/data): config group of config files for datasets
    - [`ssl_model`](empirical_evaluation/hydra_configs/ssl_model): config group of config files for self-supervised learning models as backbones
    - [`experiment.yaml`](empirical_evaluation/hydra_configs/experiment.yaml): config file to define the 
      architecture(s), dataset, and multi-annotator classification approach for an experiment
  - [`jupyter_notebooks`](empirical_evaluation/jupyter_notebooks): Jupyter notebooks to analyze results
    - [`tabular_results.ipynb`](empirical_evaluation/jupyter_notebooks/tabular_results.ipynb): Jupyter notebook to create the tables of results obtained after executing the 
      experiments for the dataset [`dopanim`](https://doi.org/10.5281/zenodo.11479590) 
  - [`python_scripts`](empirical_evaluation/python_scripts): collection of scripts to perform experimental evaluation
    - [`perform_experiments.py`](empirical_evaluation/python_scripts/perform_experiments.py): script to execute a single experiment for a given configuration
    - [`write_bash_scripts.py`](empirical_evaluation/python_scripts/write_bash_scripts.py): script to write Bash or Slurm scripts for evaluation
- [`maml`](maml): Python package for multi-annotator machine learning consisting of several sub-packages
    - [`architectures`](maml/architectures): implementations of network architectures for the ground truth and 
      annotator performance models
    - [`classifiers`](maml/classifiers): implementations of multi-annotator machine learning approaches using 
      [`pytorch_lightning`](https://www.pytorchlightning.ai/) modules
    - [`data`](maml/data): implementations of [`pytorch`](https://pytorch.org/) data sets with class labels provided by multiple,
      error-prone annotators
    - [`utils`](maml/utils): helper functions, e.g., for visualization
- [`environment.yml`](environment.yml): file containing all package details to create a 
  [`conda`](https://conda.io/projects/conda/en/latest/) environment


## Trouble Shooting :rotating_light:
If you encounter any problems, watch out for any `TODO` comments, which give hints or instructions to ensure the 
code's functionality. If the problems are still not resolved, feel free to create a corresponding GitHub issue
or contact us directly via the e-mail [marek.herde@uni-kassel.de](mailto:marek.herde@uni-kassel.de)
