"""Backwards-compatibility shim — use data_complexity.model_experiments.experiment.run instead."""
from data_complexity.model_experiments.experiment.run import *  # noqa: F401, F403
from data_complexity.model_experiments.experiment.run import (
    Experiment,
    _run_param_value_worker,
)
# Re-export utils symbols that were previously imported from this module
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
