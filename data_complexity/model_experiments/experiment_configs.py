"""Backwards-compatibility shim — use data_complexity.model_experiments.experiment.config instead."""
from data_complexity.model_experiments.experiment.config import *  # noqa: F401, F403
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
