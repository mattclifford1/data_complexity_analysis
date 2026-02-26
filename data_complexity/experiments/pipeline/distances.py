"""
Distance computation for experiment results.

Standalone functions that accept an ``Experiment`` instance as their first
argument — following the same delegation pattern as runner.py, plotting.py, and io.py.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

from data_complexity.experiments.pipeline.metric_distance import (
    DistanceBetweenMetrics,
    PearsonCorrelation,
)
from data_complexity.experiments.pipeline.utils import RunMode

if TYPE_CHECKING:
    from data_complexity.experiments.pipeline.experiment import Experiment


def compute_distances(
    experiment: "Experiment",
    ml_column: Optional[str] = None,
    complexity_source: str = "train",
    ml_source: str = "test",
    distance: DistanceBetweenMetrics = PearsonCorrelation(),
) -> pd.DataFrame:
    """
    Compute distances between complexity metrics and ML performance.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    ml_column : str, optional
        ML metric column to measure against. Default: config.correlation_target
    complexity_source : str
        Which complexity to use: 'train' or 'test'. Default: 'train'
    ml_source : str
        Which ML results to use: 'train' or 'test'. Default: 'test'
    distance : DistanceBetweenMetrics
        Distance/association measure to use. Default: PearsonCorrelation()

    Returns
    -------
    pd.DataFrame
        Results sorted by abs_distance descending.
        Columns: complexity_metric, ml_metric, distance, p_value, abs_distance.
        p_value is NaN when the measure does not produce a significance test.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing distances.")

    run_mode = experiment.config.run_mode
    if run_mode == RunMode.COMPLEXITY_ONLY:
        raise RuntimeError(
            "Cannot compute distances with run_mode=COMPLEXITY_ONLY (no ML results)."
        )
    if run_mode == RunMode.ML_ONLY:
        raise RuntimeError(
            "Cannot compute distances with run_mode=ML_ONLY (no complexity results)."
        )

    ml_column = ml_column or experiment.config.correlation_target
    complexity_df = experiment.results._get_complexity_df(complexity_source)
    ml_df = experiment.results._get_ml_df(ml_source)

    metric_cols = [
        c for c in complexity_df.columns if c not in ("param_value", "param_label")
    ]
    ml_values = ml_df[ml_column].values
    results = []

    if np.std(ml_values) == 0:
        distances_df = pd.DataFrame(
            columns=["complexity_metric", "ml_metric", "distance", "p_value", "abs_distance"]
        )
        experiment.results.distances_df = distances_df
        return distances_df

    for metric in metric_cols:
        values = complexity_df[metric].values

        if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (NearConstantInputWarning, ConstantInputWarning))
            d, p = distance.compute(values, ml_values)
        results.append(
            {
                "complexity_metric": metric,
                "ml_metric": ml_column,
                "distance": d,
                "p_value": p if p is not None else float("nan"),
                "abs_distance": abs(d),
            }
        )

    distances_df = pd.DataFrame(results).sort_values("abs_distance", ascending=False)
    experiment.results.distances_df = distances_df
    return distances_df


def compute_complexity_pairwise_distances(
    experiment: "Experiment",
    source: str = "train",
    distances: Optional[List[DistanceBetweenMetrics]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute pairwise distances between complexity metrics for all requested measures.

    Always computes for both train and test data (when available).
    Results stored in ``results.complexity_pairwise_distances`` and
    ``results.complexity_pairwise_distances_test``, keyed by measure name.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    source : str
        Which dict to return: 'train' or 'test'. Default: 'train'
    distances : list of DistanceBetweenMetrics, optional
        Measures to compute. Default: config.pairwise_distance_measures

    Returns
    -------
    dict
        name -> N×N symmetric DataFrame for the requested source.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing distances.")

    measures = distances or experiment.config.pairwise_distance_measures

    def _compute_for_source(src: str, measure: DistanceBetweenMetrics) -> Optional[pd.DataFrame]:
        complexity_df = experiment.results._get_complexity_df(src)
        if complexity_df is None:
            return None
        metric_cols = [
            c for c in complexity_df.columns
            if c not in ("param_value", "param_label") and not c.endswith("_std")
        ]
        if not metric_cols:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (NearConstantInputWarning, ConstantInputWarning))
            return complexity_df[metric_cols].corr(
                method=lambda x, y: measure.compute(x, y)[0]
            )

    for measure in measures:
        train_mat = _compute_for_source("train", measure)
        if train_mat is not None:
            experiment.results.complexity_pairwise_distances[measure.name] = train_mat

        test_mat = _compute_for_source("test", measure)
        if test_mat is not None:
            experiment.results.complexity_pairwise_distances_test[measure.name] = test_mat

    return (
        experiment.results.complexity_pairwise_distances
        if source == "train"
        else experiment.results.complexity_pairwise_distances_test
    )


def compute_ml_pairwise_distances(
    experiment: "Experiment",
    source: str = "test",
    distances: Optional[List[DistanceBetweenMetrics]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute pairwise distances between ML performance metrics for all requested measures.

    Results stored in ``results.ml_pairwise_distances``, keyed by measure name.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    source : str
        Which ML data to use: 'train' or 'test'. Default: 'test'
    distances : list of DistanceBetweenMetrics, optional
        Measures to compute. Default: config.pairwise_distance_measures

    Returns
    -------
    dict
        name -> N×N symmetric DataFrame.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing distances.")
    if experiment.config.run_mode == RunMode.COMPLEXITY_ONLY:
        raise RuntimeError(
            "Cannot compute ML distances with run_mode=COMPLEXITY_ONLY (no ML results)."
        )

    measures = distances or experiment.config.pairwise_distance_measures
    ml_df = experiment.results._get_ml_df(source)
    metric_cols = [
        c for c in ml_df.columns
        if c != "param_value" and not c.endswith("_std")
    ]
    valid_cols = [c for c in metric_cols if ml_df[c].std() > 0]

    for measure in measures:
        distance_matrix = ml_df[valid_cols].corr(
            method=lambda x, y: measure.compute(x, y)[0]
        )
        experiment.results.ml_pairwise_distances[measure.name] = distance_matrix

    return experiment.results.ml_pairwise_distances


def compute_all_distances(
    experiment: "Experiment",
    distance: DistanceBetweenMetrics = PearsonCorrelation(),
) -> pd.DataFrame:
    """
    Compute distances for all ML metric columns.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    distance : DistanceBetweenMetrics
        Distance/association measure to use. Default: PearsonCorrelation()

    Returns
    -------
    pd.DataFrame
        All distances combined.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing distances.")

    ml_df = experiment.results.ml_df
    ml_cols = [c for c in ml_df.columns if c != "param_value"]

    all_distances = []
    for ml_col in ml_cols:
        dist_df = _compute_single_distance(experiment, ml_col, distance=distance)
        all_distances.append(dist_df)

    return pd.concat(all_distances, ignore_index=True)


def compute_per_classifier_distances(
    experiment: "Experiment",
    complexity_source: str = "train",
    ml_source: str = "test",
    distance: DistanceBetweenMetrics = PearsonCorrelation(),
) -> pd.DataFrame:
    """
    Compute complexity-vs-ML distances aggregated per metric across classifiers.

    For each ML metric type (e.g., 'accuracy', 'f1'), computes the distance
    between each complexity metric and every individual classifier's column,
    then reports mean and std of those values.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    complexity_source : str
        'train' or 'test'. Default: 'train'
    ml_source : str
        'train' or 'test'. Default: 'test'
    distance : DistanceBetweenMetrics
        Distance/association measure to use. Default: PearsonCorrelation()

    Returns
    -------
    pd.DataFrame
        Columns: complexity_metric, ml_metric, mean_distance,
                 std_distance, abs_mean_distance.
        Sorted by abs_mean_distance descending.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing distances.")

    complexity_df = experiment.results._get_complexity_df(complexity_source)
    ml_df = experiment.results._get_ml_df(ml_source)

    complexity_cols = [
        c for c in complexity_df.columns
        if c not in ("param_value", "param_label") and not c.endswith("_std")
    ]

    rows = []
    for metric_name in experiment.config.ml_metrics:
        classifier_cols = [
            c for c in ml_df.columns
            if c.endswith(f"_{metric_name}")
            and not c.startswith("best_")
            and not c.startswith("mean_")
            and not c.endswith("_std")
        ]

        for complexity_col in complexity_cols:
            comp_values = complexity_df[complexity_col].values
            if np.std(comp_values) == 0 or np.any(np.isnan(comp_values)):
                continue

            ds = []
            for clf_col in classifier_cols:
                ml_values = ml_df[clf_col].values
                if np.std(ml_values) == 0 or np.any(np.isnan(ml_values)):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", (NearConstantInputWarning, ConstantInputWarning)
                    )
                    d, _ = distance.compute(comp_values, ml_values)
                if not np.isnan(d):
                    ds.append(d)

            if not ds:
                continue

            mean_d = float(np.mean(ds))
            rows.append(
                {
                    "complexity_metric": complexity_col,
                    "ml_metric": metric_name,
                    "mean_distance": mean_d,
                    "std_distance": float(np.std(ds)),
                    "abs_mean_distance": abs(mean_d),
                }
            )

    df = pd.DataFrame(rows).sort_values("abs_mean_distance", ascending=False)
    experiment.results.per_classifier_distances_df = df
    return df


def _compute_single_distance(
    experiment: "Experiment",
    ml_column: str,
    distance: DistanceBetweenMetrics = PearsonCorrelation(),
) -> pd.DataFrame:
    """Compute distances for a single ML column."""
    complexity_df = experiment.results.complexity_df
    ml_df = experiment.results.ml_df

    metric_cols = [
        c for c in complexity_df.columns if c not in ("param_value", "param_label")
    ]
    ml_values = ml_df[ml_column].values
    results = []

    if np.std(ml_values) == 0:
        return pd.DataFrame(
            columns=["complexity_metric", "ml_metric", "distance", "p_value", "abs_distance"]
        )

    for metric in metric_cols:
        values = complexity_df[metric].values

        if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (NearConstantInputWarning, ConstantInputWarning))
            d, p = distance.compute(values, ml_values)
        results.append(
            {
                "complexity_metric": metric,
                "ml_metric": ml_column,
                "distance": d,
                "p_value": p if p is not None else float("nan"),
                "abs_distance": abs(d),
            }
        )

    if not results:
        raise ValueError("No valid distances computed.")

    return pd.DataFrame(results).sort_values("abs_distance", ascending=False)
