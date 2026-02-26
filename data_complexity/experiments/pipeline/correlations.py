"""
Correlation computation for experiment results.

Standalone functions that accept an ``Experiment`` instance as their first
argument — following the same delegation pattern as runner.py, plotting.py, and io.py.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

from data_complexity.experiments.pipeline.utils import RunMode

if TYPE_CHECKING:
    from data_complexity.experiments.pipeline.experiment import Experiment


def compute_correlations(
    experiment: "Experiment",
    ml_column: Optional[str] = None,
    complexity_source: str = "train",
    ml_source: str = "test",
) -> pd.DataFrame:
    """
    Compute correlations between complexity metrics and ML performance.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    ml_column : str, optional
        ML metric column to correlate against. Default: config.correlation_target
    complexity_source : str
        Which complexity to use: 'train' or 'test'. Default: 'train'
    ml_source : str
        Which ML results to use: 'train' or 'test'. Default: 'test'

    Returns
    -------
    pd.DataFrame
        Correlation results sorted by absolute correlation.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing correlations.")

    run_mode = experiment.config.run_mode
    if run_mode == RunMode.COMPLEXITY_ONLY:
        raise RuntimeError(
            "Cannot compute correlations with run_mode=COMPLEXITY_ONLY (no ML results)."
        )
    if run_mode == RunMode.ML_ONLY:
        raise RuntimeError(
            "Cannot compute correlations with run_mode=ML_ONLY (no complexity results)."
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
        correlations_df = pd.DataFrame(
            columns=["complexity_metric", "ml_metric", "correlation", "p_value", "abs_correlation"]
        )
        experiment.results.correlations_df = correlations_df
        return correlations_df

    for metric in metric_cols:
        values = complexity_df[metric].values

        if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (NearConstantInputWarning, ConstantInputWarning))
            r, p = stats.pearsonr(values, ml_values)
        results.append(
            {
                "complexity_metric": metric,
                "ml_metric": ml_column,
                "correlation": r,
                "p_value": p,
                "abs_correlation": abs(r),
            }
        )

    correlations_df = pd.DataFrame(results).sort_values("abs_correlation", ascending=False)
    experiment.results.correlations_df = correlations_df
    return correlations_df


def compute_complexity_correlations(
    experiment: "Experiment",
    source: str = "train",
) -> Optional[pd.DataFrame]:
    """
    Compute pairwise Pearson correlations between complexity metrics.

    Always computes correlations for both train and test data (when available).
    Train correlations are stored in ``results.complexity_correlations_df`` (backward compat);
    test correlations are stored in ``results.complexity_correlations_test_df``.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    source : str
        Which correlation matrix to return: 'train' or 'test'. Default: 'train'

    Returns
    -------
    pd.DataFrame or None
        N×N symmetric correlation matrix for the requested source.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing correlations.")

    def _compute_for_source(src: str) -> Optional[pd.DataFrame]:
        complexity_df = experiment.results._get_complexity_df(src)
        if complexity_df is None:
            return None
        metric_cols = [
            c for c in complexity_df.columns
            if c not in ("param_value", "param_label") and not c.endswith("_std")
        ]
        valid_cols = [c for c in metric_cols if complexity_df[c].std() > 0]
        if not valid_cols:
            return None
        return complexity_df[valid_cols].corr(method="pearson")

    train_corr = _compute_for_source("train")
    if train_corr is not None:
        experiment.results.complexity_correlations_df = train_corr

    test_corr = _compute_for_source("test")
    if test_corr is not None:
        experiment.results.complexity_correlations_test_df = test_corr

    return train_corr if source == "train" else test_corr


def compute_ml_correlations(
    experiment: "Experiment",
    source: str = "test",
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations between ML performance metrics.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    source : str
        Which ML data to use: 'train' or 'test'. Default: 'test'

    Returns
    -------
    pd.DataFrame
        N×N symmetric correlation matrix indexed and columned by metric names.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing correlations.")
    if experiment.config.run_mode == RunMode.COMPLEXITY_ONLY:
        raise RuntimeError(
            "Cannot compute ML correlations with run_mode=COMPLEXITY_ONLY (no ML results)."
        )

    ml_df = experiment.results._get_ml_df(source)
    metric_cols = [
        c for c in ml_df.columns
        if c != "param_value" and not c.endswith("_std")
    ]
    # Drop constant metrics (zero std) — they produce NaN correlations
    valid_cols = [c for c in metric_cols if ml_df[c].std() > 0]

    corr_matrix = ml_df[valid_cols].corr(method="pearson")
    experiment.results.ml_correlations_df = corr_matrix
    return corr_matrix


def compute_all_correlations(experiment: "Experiment") -> pd.DataFrame:
    """
    Compute correlations for all ML metric columns.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.

    Returns
    -------
    pd.DataFrame
        All correlations combined.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing correlations.")

    ml_df = experiment.results.ml_df
    ml_cols = [c for c in ml_df.columns if c != "param_value"]

    all_corrs = []
    for ml_col in ml_cols:
        corr_df = _compute_single_correlation(experiment, ml_col)
        all_corrs.append(corr_df)

    return pd.concat(all_corrs, ignore_index=True)


def compute_per_classifier_correlations(
    experiment: "Experiment",
    complexity_source: str = "train",
    ml_source: str = "test",
) -> pd.DataFrame:
    """
    Compute complexity-vs-ML correlations aggregated per metric across classifiers.

    For each ML metric type (e.g., 'accuracy', 'f1'), computes the Pearson
    correlation between each complexity metric and every individual classifier's
    column, then reports mean and std of those correlations.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance holding config and results.
    complexity_source : str
        'train' or 'test'. Default: 'train'
    ml_source : str
        'train' or 'test'. Default: 'test'

    Returns
    -------
    pd.DataFrame
        Columns: complexity_metric, ml_metric, mean_correlation,
                 std_correlation, abs_mean_correlation.
        Sorted by abs_mean_correlation descending.
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before computing correlations.")

    complexity_df = experiment.results._get_complexity_df(complexity_source)
    ml_df = experiment.results._get_ml_df(ml_source)

    complexity_cols = [
        c for c in complexity_df.columns
        if c not in ("param_value", "param_label") and not c.endswith("_std")
    ]

    rows = []
    for metric_name in experiment.config.ml_metrics:
        # Per-classifier columns: end with _{metric_name}, no best_/mean_ prefix, no _std suffix
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

            rs = []
            for clf_col in classifier_cols:
                ml_values = ml_df[clf_col].values
                if np.std(ml_values) == 0 or np.any(np.isnan(ml_values)):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", (NearConstantInputWarning, ConstantInputWarning)
                    )
                    r, _ = stats.pearsonr(comp_values, ml_values)
                if not np.isnan(r):
                    rs.append(r)

            if not rs:
                continue

            mean_r = float(np.mean(rs))
            rows.append(
                {
                    "complexity_metric": complexity_col,
                    "ml_metric": metric_name,
                    "mean_correlation": mean_r,
                    "std_correlation": float(np.std(rs)),
                    "abs_mean_correlation": abs(mean_r),
                }
            )

    df = pd.DataFrame(rows).sort_values("abs_mean_correlation", ascending=False)
    experiment.results.per_classifier_correlations_df = df
    return df


def _compute_single_correlation(experiment: "Experiment", ml_column: str) -> pd.DataFrame:
    """Compute correlations for a single ML column."""
    complexity_df = experiment.results.complexity_df
    ml_df = experiment.results.ml_df

    metric_cols = [
        c for c in complexity_df.columns if c not in ("param_value", "param_label")
    ]
    ml_values = ml_df[ml_column].values
    results = []

    if np.std(ml_values) == 0:
        return pd.DataFrame(
            columns=["complexity_metric", "ml_metric", "correlation", "p_value", "abs_correlation"]
        )

    for metric in metric_cols:
        values = complexity_df[metric].values

        if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (NearConstantInputWarning, ConstantInputWarning))
            r, p = stats.pearsonr(values, ml_values)
        results.append(
            {
                "complexity_metric": metric,
                "ml_metric": ml_column,
                "correlation": r,
                "p_value": p,
                "abs_correlation": abs(r),
            }
        )

    if not results:
        raise ValueError("No valid correlations computed.")

    return pd.DataFrame(results).sort_values("abs_correlation", ascending=False)
