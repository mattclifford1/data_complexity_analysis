"""
Experiment package for complexity vs ML performance analysis.

Sub-modules
-----------
utils   : Data classes, results container, and helper functions.
run     : Experiment runner (Experiment class) and parallel worker.
config  : Pre-defined experiment configurations.
"""
from data_complexity.model_experiments.experiment.utils import (
    PlotType,
    ParameterSpec,
    DatasetSpec,
    ExperimentConfig,
    ExperimentResultsContainer,
    _average_dicts,
    _average_ml_results,
    _std_dicts,
    make_json_safe_dict,
    make_json_safe_list,
)
from data_complexity.model_experiments.experiment.run import (
    Experiment,
    _run_param_value_worker,
)
from data_complexity.model_experiments.experiment.config import (
    EXPERIMENT_CONFIGS,
    gaussian_variance_config,
    gaussian_separation_config,
    gaussian_correlation_config,
    gaussian_imbalance_config,
    moons_noise_config,
    circles_noise_config,
    blobs_features_config,
    get_config,
    list_configs,
    run_experiment,
    run_all_experiments,
)

__all__ = [
    # utils
    "PlotType",
    "ParameterSpec",
    "DatasetSpec",
    "ExperimentConfig",
    "ExperimentResultsContainer",
    "_average_dicts",
    "_average_ml_results",
    "_std_dicts",
    "make_json_safe_dict",
    "make_json_safe_list",
    # run
    "Experiment",
    "_run_param_value_worker",
    # config
    "EXPERIMENT_CONFIGS",
    "gaussian_variance_config",
    "gaussian_separation_config",
    "gaussian_correlation_config",
    "gaussian_imbalance_config",
    "moons_noise_config",
    "circles_noise_config",
    "blobs_features_config",
    "get_config",
    "list_configs",
    "run_experiment",
    "run_all_experiments",
]
