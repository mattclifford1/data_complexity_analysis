"""
Experiment package for complexity vs ML performance analysis.

Sub-modules
-----------
utils   : Data classes, results container, and helper functions.
run     : Experiment runner (Experiment class) and parallel worker.
config  : Pre-defined experiment configurations.
"""
from data_complexity.experiments.pipeline.metric_distance import (
    DistanceBetweenMetrics,
    PearsonCorrelation,
    SpearmanCorrelation,
    KendallTau,
    MutualInformation,
    EuclideanDistance,
)
from data_complexity.experiments.pipeline.utils import (
    PlotType,
    RunMode,
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
from data_complexity.experiments.pipeline.experiment import Experiment
from data_complexity.experiments.pipeline.runner import (
    _run_dataset_spec_worker,
)
from data_complexity.experiments.pipeline.legacy_complexity_over_datasets import (
    ComplexityCollection,
    DatasetEntry,
)
from data_complexity.experiments.pipeline.config import (
    EXPERIMENT_CONFIGS,
    datasets_from_sweep,
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
    # metric_distance
    "DistanceBetweenMetrics",
    "PearsonCorrelation",
    "SpearmanCorrelation",
    "KendallTau",
    "MutualInformation",
    "EuclideanDistance",
    # utils
    "PlotType",
    "RunMode",
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
    "_run_dataset_spec_worker",
    "ComplexityCollection",
    "DatasetEntry",
    # config
    "EXPERIMENT_CONFIGS",
    "datasets_from_sweep",
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
