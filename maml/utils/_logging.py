from mlflow import log_param
from omegaconf import DictConfig, ListConfig


def log_params_from_omegaconf_dict(params: dict):
    """
    Logs the parameters in a dictionary via `mlflow`.

    Parameters
    ----------
    params : dict
        Dictionary of parameters to be logged via `mlflow`.
    """
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            log_param(f"{parent_name}.{i}", v)
    else:
        log_param(f"{parent_name}", element)
