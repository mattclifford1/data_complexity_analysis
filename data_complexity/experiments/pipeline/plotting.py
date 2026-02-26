"""
Plotting logic for the Experiment class.

This module contains the plot() function that generates all experiment visualizations.
It is designed to be called from Experiment.plot() via thin delegation.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from data_complexity.experiments.pipeline.utils import PlotType, RunMode
from data_complexity.experiments.plotting import (
    plot_correlations,
    plot_summary,
    plot_model_comparison,
    plot_metrics_vs_parameter,
    plot_models_vs_parameter,
    plot_complexity_metrics_vs_parameter,
    plot_models_vs_parameter_combined,
    plot_complexity_metrics_vs_parameter_combined,
    plot_datasets_overview,
    plot_complexity_correlations_heatmap,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from data_complexity.experiments.pipeline.experiment import Experiment


def plot(
    experiment: "Experiment",
    plot_types: Optional[List[PlotType]] = None,
) -> Dict[Union[PlotType, str], "plt.Figure"]:
    """
    Generate experiment plots.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance providing results, config, and datasets.
    plot_types : list of PlotType, optional
        Plot types to generate. Default: config.plots

    Returns
    -------
    dict
        PlotType or str -> matplotlib Figure. String keys are used when a single
        PlotType generates multiple figures (e.g. COMPLEXITY_CORRELATIONS produces
        'complexity_correlations_train' and 'complexity_correlations_test').
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before plotting.")

    plot_types = plot_types or experiment.config.plots
    figures = {}
    run_mode = experiment.config.run_mode

    if run_mode == RunMode.BOTH and experiment.results.correlations_df is None:
        experiment.compute_correlations()

    if PlotType.COMPLEXITY_CORRELATIONS in plot_types and experiment.results.complexity_correlations_df is None:
        experiment.compute_complexity_correlations()

    if PlotType.ML_CORRELATIONS in plot_types and experiment.results.ml_correlations_df is None:
        if experiment.config.run_mode != RunMode.COMPLEXITY_ONLY:
            experiment.compute_ml_correlations()

    for pt in plot_types:
        if pt == PlotType.CORRELATIONS:
            if run_mode != RunMode.BOTH:
                warnings.warn(
                    f"Skipping {pt.name} plot: requires both complexity and ML results "
                    f"(run_mode={run_mode.value})."
                )
                continue
            fig = plot_correlations(
                experiment.results.correlations_df,
                title=f"{experiment.config.name}: Complexity vs {experiment.config.correlation_target}",
            )
            figures[pt] = fig

        elif pt == PlotType.SUMMARY:
            if run_mode != RunMode.BOTH:
                warnings.warn(
                    f"Skipping {pt.name} plot: requires both complexity and ML results "
                    f"(run_mode={run_mode.value})."
                )
                continue
            fig = plot_summary(
                experiment.results.complexity_df,
                experiment.results.ml_df,
                experiment.results.correlations_df,
                ml_column=experiment.config.correlation_target,
                top_n=6,
            )
            figures[pt] = fig

        elif pt == PlotType.HEATMAP:
            if run_mode != RunMode.BOTH:
                warnings.warn(
                    f"Skipping {pt.name} plot: requires both complexity and ML results "
                    f"(run_mode={run_mode.value})."
                )
                continue
            all_corr = experiment.compute_all_correlations()
            fig = plot_model_comparison(all_corr)
            figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_TRAIN:
            if experiment.results.train_complexity_df is not None and experiment.results.train_ml_df is not None:
                fig = plot_metrics_vs_parameter(
                    complexity_df=experiment.results.train_complexity_df,
                    ml_df=experiment.results.train_ml_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Train — metrics vs {experiment.config.x_label}",
                )
                figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_TEST:
            if experiment.results.test_complexity_df is not None and experiment.results.test_ml_df is not None:
                fig = plot_metrics_vs_parameter(
                    complexity_df=experiment.results.test_complexity_df,
                    ml_df=experiment.results.test_ml_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Test — metrics vs {experiment.config.x_label}",
                )
                figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_MODELS_TRAIN:
            if experiment.results.train_ml_df is not None:
                fig = plot_models_vs_parameter(
                    ml_df=experiment.results.train_ml_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Models (Train) vs {experiment.config.x_label}",
                    ml_metrics=experiment.config.ml_metrics,
                )
                figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_MODELS_TEST:
            if experiment.results.test_ml_df is not None:
                fig = plot_models_vs_parameter(
                    ml_df=experiment.results.test_ml_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Models (Test) vs {experiment.config.x_label}",
                    ml_metrics=experiment.config.ml_metrics,
                )
                figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_COMPLEXITY_TRAIN:
            if experiment.results.train_complexity_df is not None:
                fig = plot_complexity_metrics_vs_parameter(
                    complexity_df=experiment.results.train_complexity_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Complexity (Train) vs {experiment.config.x_label}",
                )
                figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_COMPLEXITY_TEST:
            if experiment.results.test_complexity_df is not None:
                fig = plot_complexity_metrics_vs_parameter(
                    complexity_df=experiment.results.test_complexity_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Complexity (Test) vs {experiment.config.x_label}",
                )
                figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_MODELS_COMBINED:
            if experiment.results.train_ml_df is not None and experiment.results.test_ml_df is not None:
                fig = plot_models_vs_parameter_combined(
                    train_ml_df=experiment.results.train_ml_df,
                    test_ml_df=experiment.results.test_ml_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Models (Train vs Test) vs {experiment.config.x_label}",
                    ml_metrics=experiment.config.ml_metrics,
                )
                figures[pt] = fig

        elif pt == PlotType.LINE_PLOT_COMPLEXITY_COMBINED:
            if experiment.results.train_complexity_df is not None and experiment.results.test_complexity_df is not None:
                fig = plot_complexity_metrics_vs_parameter_combined(
                    train_complexity_df=experiment.results.train_complexity_df,
                    test_complexity_df=experiment.results.test_complexity_df,
                    param_label_col="param_value",
                    x_label=experiment.config.x_label,
                    title=f"{experiment.config.name}: Complexity (Train vs Test) vs {experiment.config.x_label}",
                )
                figures[pt] = fig

        elif pt == PlotType.DATASETS_OVERVIEW:
            if experiment.datasets:
                fig = plot_datasets_overview(
                    datasets=experiment.datasets,
                    format_label=lambda label: label,
                )
                figures[pt] = fig

        elif pt == PlotType.COMPLEXITY_CORRELATIONS:
            if experiment.results.complexity_correlations_df is None:
                experiment.compute_complexity_correlations()
            if experiment.results.complexity_correlations_df is not None:
                fig_train = plot_complexity_correlations_heatmap(
                    experiment.results.complexity_correlations_df,
                    title=f"{experiment.config.name}: Complexity Metric Correlations (Train)",
                )
                figures["complexity_correlations_train"] = fig_train
            if experiment.results.complexity_correlations_test_df is not None:
                fig_test = plot_complexity_correlations_heatmap(
                    experiment.results.complexity_correlations_test_df,
                    title=f"{experiment.config.name}: Complexity Metric Correlations (Test)",
                )
                figures["complexity_correlations_test"] = fig_test

        elif pt == PlotType.ML_CORRELATIONS:
            if experiment.config.run_mode == RunMode.COMPLEXITY_ONLY:
                warnings.warn(
                    f"Skipping {pt.name} plot: requires ML results "
                    f"(run_mode={experiment.config.run_mode.value})."
                )
                continue
            if experiment.results.ml_correlations_df is None:
                experiment.compute_ml_correlations()
            fig = plot_complexity_correlations_heatmap(
                experiment.results.ml_correlations_df,
                title=f"{experiment.config.name}: ML Metric Correlations",
            )
            figures[pt] = fig

    return figures
