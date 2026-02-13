"""
Experiment utilities for complexity vs ML experiments.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from data_complexity.model_experiments.ml import (
    AbstractMLModel,
    get_default_models,
    evaluate_models_train_test,
    get_best_metric,
    get_mean_metric,
    get_metrics_from_names,
)


class PlotType(Enum):
    """Types of plots that can be generated."""

    CORRELATIONS = auto()
    METRIC_VS_ACCURACY = auto()
    SUMMARY = auto()
    HEATMAP = auto()


@dataclass
class ParameterSpec:
    """
    Specification for a parameter to vary in an experiment.

    Parameters
    ----------
    name : str
        Parameter name as passed to get_dataset().
    values : list
        Values to iterate over.
    label_format : str
        Format string for labeling data points. Use {name} and {value} placeholders.
    """

    name: str
    values: List[Any]
    label_format: str = "{name}={value}"

    def format_label(self, value: Any) -> str:
        """Format a label for a specific parameter value."""
        return self.label_format.format(name=self.name, value=value)


@dataclass
class DatasetSpec:
    """
    Specification for the dataset type and fixed parameters.

    Parameters
    ----------
    dataset_type : str
        Type of dataset ('Gaussian', 'Moons', 'Circles', 'Blobs', 'XOR').
    fixed_params : dict
        Parameters that remain constant across experiment iterations.
    """

    dataset_type: str
    # fixed_params can include any parameters accepted by the dataset generation functions
    # e.g.: for Gaussian: class_separation, cov_type, cov_scale, equal_test
    #     or train_size, num_samples (if we want to fix those instead of varying them)
    fixed_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """
    Configuration for a complexity vs ML experiment.

    Parameters
    ----------
    dataset : DatasetSpec
        Dataset specification.
    vary_parameter : ParameterSpec
        Parameter to vary across experiment iterations.
    models : list of AbstractMLModel, optional
        ML models to evaluate. Default: get_default_models()
    ml_metrics : list of str
        ML metrics to compute. Default: ['accuracy', 'f1']
    cv_folds : int
        Number of random seeds for train/test splitting. Default: 5
    name : str, optional
        Experiment name. Auto-generated if None.
    save_dir : Path, optional
        Directory to save results. Default: results/{name}/
    plots : list of PlotType
        Plot types to generate. Default: [CORRELATIONS, SUMMARY]
    correlation_target : str
        ML metric to correlate against. Default: 'best_accuracy'
    train_post_process : Callable, optional
        Applied to train data after the train/test split and before complexity
        computation and ML training. Takes a data dict ``{"X": np.ndarray,
        "y": np.ndarray}`` and returns a (possibly new) data dict. ``None``
        means no transformation.
    test_post_process : Callable, optional
        Applied to test data after the train/test split and before complexity
        computation and ML evaluation. Same signature as ``train_post_process``.
        ``None`` means no transformation.
    """

    dataset: DatasetSpec
    vary_parameter: ParameterSpec
    models: Optional[List[AbstractMLModel]] = None
    ml_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1"])
    cv_folds: int = 5
    name: Optional[str] = None
    save_dir: Optional[Path] = None
    plots: List[PlotType] = field(
        default_factory=lambda: [PlotType.CORRELATIONS, PlotType.SUMMARY]
    )
    correlation_target: str = "best_accuracy"
    train_post_process: Optional[Callable[[Dict], Dict]] = None
    test_post_process: Optional[Callable[[Dict], Dict]] = None
    equal_test: bool = False # If True, ensures test set is balanced for imbalance experiments

    def __post_init__(self):
        """Generate name and save_dir if not provided."""
        if self.name is None:
            self.name = self._generate_name()
        if self.save_dir is None:
            self.save_dir = Path(__file__).parent / "runs" / "results" / self.name

    def _generate_name(self) -> str:
        """Generate experiment name from dataset type and varied parameter."""
        return f"{self.dataset.dataset_type.lower()}_{self.vary_parameter.name}"


def _average_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of metric dicts element-wise."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: np.mean([d[k] for d in dicts if k in d]) for k in keys}


def _std_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute std of a list of metric dicts element-wise."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: np.std([d[k] for d in dicts if k in d]) for k in keys}


def _average_ml_results(
    results_list: List[Dict[str, Dict[str, Dict[str, float]]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Average ML results across multiple seeds.

    Parameters
    ----------
    results_list : list
        Each element: model_name -> metric_name -> {'mean': float, 'std': float}

    Returns
    -------
    dict
        Averaged results with std computed across seeds.
    """
    if not results_list:
        return {}

    model_names = results_list[0].keys()
    averaged = {}

    for model in model_names:
        averaged[model] = {}
        metric_names = results_list[0][model].keys()
        for metric in metric_names:
            values = [r[model][metric]["mean"] for r in results_list if model in r and metric in r[model]]
            averaged[model][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }

    return averaged


class ExperimentResultsContainer:
    """
    Container for experiment results with DataFrame storage.

    Stores complexity metrics and ML performance for each parameter value,
    separately for train and test splits. For backward compatibility,
    ``complexity_df`` returns train complexity and ``ml_df`` returns test ML.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # Legacy rows (kept for backward compat with add_result)
        self._complexity_rows: List[Dict[str, Any]] = []
        self._ml_rows: List[Dict[str, Any]] = []
        self._complexity_df: Optional[pd.DataFrame] = None
        self._ml_df: Optional[pd.DataFrame] = None
        self._correlations_df: Optional[pd.DataFrame] = None

        # Train/test split rows
        self._train_complexity_rows: List[Dict[str, Any]] = []
        self._test_complexity_rows: List[Dict[str, Any]] = []
        self._train_ml_rows: List[Dict[str, Any]] = []
        self._test_ml_rows: List[Dict[str, Any]] = []
        self._train_complexity_df: Optional[pd.DataFrame] = None
        self._test_complexity_df: Optional[pd.DataFrame] = None
        self._train_ml_df: Optional[pd.DataFrame] = None
        self._test_ml_df: Optional[pd.DataFrame] = None

    def _build_ml_row(
        self,
        param_value: Any,
        ml_results: Dict[str, Dict[str, Dict[str, float]]],
    ) -> Dict[str, Any]:
        """Build an ML row dict from model results."""
        ml_row: Dict[str, Any] = {"param_value": param_value}
        for metric in self.config.ml_metrics:
            ml_row[f"best_{metric}"] = get_best_metric(ml_results, metric)
            ml_row[f"mean_{metric}"] = get_mean_metric(ml_results, metric)
        for model_name, metrics in ml_results.items():
            for metric in self.config.ml_metrics:
                if metric in metrics:
                    ml_row[f"{model_name}_{metric}"] = metrics[metric]["mean"]
                    ml_row[f"{model_name}_{metric}_std"] = metrics[metric]["std"]
        return ml_row

    def _build_complexity_row(
        self,
        param_value: Any,
        complexity_metrics_dict: Dict[str, float],
        std_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Build a complexity row dict, optionally including per-metric std columns."""
        row = {
            "param_value": param_value,
            "param_label": self.config.vary_parameter.format_label(param_value),
        }
        row.update(complexity_metrics_dict)
        if std_dict:
            row.update({f"{k}_std": v for k, v in std_dict.items()})
        return row

    def add_result(
        self,
        param_value: Any,
        complexity_metrics_dict: Dict[str, float],
        ml_results: Dict[str, Dict[str, Dict[str, float]]],
    ) -> None:
        """
        Add results for a single parameter value (legacy, no train/test split).

        Parameters
        ----------
        param_value : Any
            The parameter value used for this iteration.
        complexity_metrics_dict : dict
            Complexity metric name -> value.
        ml_results : dict
            Model name -> metric name -> {'mean': float, 'std': float}
        """
        self._complexity_rows.append(
            self._build_complexity_row(param_value, complexity_metrics_dict)
        )
        self._ml_rows.append(self._build_ml_row(param_value, ml_results))

    def add_split_result(
        self,
        param_value: Any,
        train_complexity_dict: Dict[str, float],
        test_complexity_dict: Dict[str, float],
        train_ml_results: Dict[str, Dict[str, Dict[str, float]]],
        test_ml_results: Dict[str, Dict[str, Dict[str, float]]],
        train_complexity_std_dict: Optional[Dict[str, float]] = None,
        test_complexity_std_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Add results for a single parameter value with train/test split.

        Parameters
        ----------
        param_value : Any
            The parameter value used for this iteration.
        train_complexity_dict : dict
            Complexity metrics computed on training data (means across seeds).
        test_complexity_dict : dict
            Complexity metrics computed on test data (means across seeds).
        train_ml_results : dict
            ML results evaluated on training data.
        test_ml_results : dict
            ML results evaluated on test data.
        train_complexity_std_dict : dict, optional
            Std of complexity metrics on training data across seeds.
        test_complexity_std_dict : dict, optional
            Std of complexity metrics on test data across seeds.
        """
        self._train_complexity_rows.append(
            self._build_complexity_row(param_value, train_complexity_dict, train_complexity_std_dict)
        )
        self._test_complexity_rows.append(
            self._build_complexity_row(param_value, test_complexity_dict, test_complexity_std_dict)
        )
        self._train_ml_rows.append(self._build_ml_row(param_value, train_ml_results))
        self._test_ml_rows.append(self._build_ml_row(param_value, test_ml_results))

    def covert_to_df(self) -> None:
        """Convert collected rows to DataFrames."""
        if self._complexity_rows:
            self._complexity_df = pd.DataFrame(self._complexity_rows)
        if self._ml_rows:
            self._ml_df = pd.DataFrame(self._ml_rows)
        if self._train_complexity_rows:
            self._train_complexity_df = pd.DataFrame(self._train_complexity_rows)
        if self._test_complexity_rows:
            self._test_complexity_df = pd.DataFrame(self._test_complexity_rows)
        if self._train_ml_rows:
            self._train_ml_df = pd.DataFrame(self._train_ml_rows)
        if self._test_ml_rows:
            self._test_ml_df = pd.DataFrame(self._test_ml_rows)

    @property
    def train_complexity_df(self) -> Optional[pd.DataFrame]:
        """Get train complexity metrics DataFrame."""
        if self._train_complexity_df is None and self._train_complexity_rows:
            self.covert_to_df()
        return self._train_complexity_df

    @property
    def test_complexity_df(self) -> Optional[pd.DataFrame]:
        """Get test complexity metrics DataFrame."""
        if self._test_complexity_df is None and self._test_complexity_rows:
            self.covert_to_df()
        return self._test_complexity_df

    @property
    def train_ml_df(self) -> Optional[pd.DataFrame]:
        """Get train ML performance DataFrame."""
        if self._train_ml_df is None and self._train_ml_rows:
            self.covert_to_df()
        return self._train_ml_df

    @property
    def test_ml_df(self) -> Optional[pd.DataFrame]:
        """Get test ML performance DataFrame."""
        if self._test_ml_df is None and self._test_ml_rows:
            self.covert_to_df()
        return self._test_ml_df

    @property
    def complexity_df(self) -> pd.DataFrame:
        """Get complexity metrics DataFrame.

        Returns train complexity if available (train/test mode),
        otherwise falls back to legacy complexity_df.
        """
        if self._train_complexity_df is not None:
            return self._train_complexity_df
        if self._complexity_df is None:
            self.covert_to_df()
        # After covert_to_df, prefer train if available
        if self._train_complexity_df is not None:
            return self._train_complexity_df
        return self._complexity_df

    @property
    def ml_df(self) -> pd.DataFrame:
        """Get ML performance DataFrame.

        Returns test ML if available (train/test mode),
        otherwise falls back to legacy ml_df.
        """
        if self._test_ml_df is not None:
            return self._test_ml_df
        if self._ml_df is None:
            self.covert_to_df()
        # After covert_to_df, prefer test if available
        if self._test_ml_df is not None:
            return self._test_ml_df
        return self._ml_df

    @property
    def correlations_df(self) -> Optional[pd.DataFrame]:
        """Get correlations DataFrame (computed separately)."""
        return self._correlations_df

    @correlations_df.setter
    def correlations_df(self, df: pd.DataFrame) -> None:
        """Set correlations DataFrame."""
        self._correlations_df = df

    def get_param_values(self) -> List[Any]:
        """Get list of parameter values."""
        return self.complexity_df["param_value"].tolist()

    def _get_complexity_df(self, source: str = "train") -> pd.DataFrame:
        """Get complexity DataFrame by source ('train' or 'test')."""
        if source == "test" and self._test_complexity_df is not None:
            return self._test_complexity_df
        if source == "train" and self._train_complexity_df is not None:
            return self._train_complexity_df
        return self.complexity_df

    def _get_ml_df(self, source: str = "test") -> pd.DataFrame:
        """Get ML DataFrame by source ('train' or 'test')."""
        if source == "train" and self._train_ml_df is not None:
            return self._train_ml_df
        if source == "test" and self._test_ml_df is not None:
            return self._test_ml_df
        return self.ml_df


def make_json_safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """get the repr of class values in a dict to make it JSON serializable (for saving config)"""
    safe_dict = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe_dict[k] = v
        else:
            if hasattr(v, "__repr__"):
                safe_dict[k] = repr(v)
            else:
                raise ValueError(f"Value for key '{k}' is not JSON serializable and has no __repr__: {v}")
    return safe_dict