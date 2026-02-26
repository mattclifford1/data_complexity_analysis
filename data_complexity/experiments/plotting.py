"""
Plotting utilities for experiment visualization.

Provides reusable plotting functions for complexity vs ML performance analysis.
"""
from typing import Any, Callable, Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_distances(
    correlations_df: pd.DataFrame,
    title: str = "Complexity vs ML Accuracy Distances",
    top_n: int = 15,
    distance_name: str = "Pearson r",
    signed: bool = True,
) -> plt.Figure:
    """
    Plot distance/correlation values as a horizontal bar chart.

    Parameters
    ----------
    correlations_df : pd.DataFrame
        DataFrame with columns: 'complexity_metric', 'distance', 'p_value'.
        Must be sorted by abs_distance descending.
    title : str
        Plot title.
    top_n : int
        Number of top values to display. Default: 15
    distance_name : str
        Label for the x-axis (e.g. 'Pearson r', 'Spearman ρ', 'Mutual Information').
    signed : bool
        If True, values range from -1 to 1; use symmetric x-axis and colour by sign.
        If False, values are non-negative; use dynamic x-axis and uniform colour.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    subset = correlations_df.head(top_n)

    metrics = subset["complexity_metric"].tolist()
    d_values = subset["distance"].tolist()
    p_values = subset["p_value"].tolist()

    if signed:
        colors = ["green" if d < 0 else "red" for d in d_values]
    else:
        colors = ["steelblue"] * len(d_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(metrics)), d_values, color=colors, alpha=0.7)

    for i, (d, p) in enumerate(zip(d_values, p_values)):
        if not pd.isna(p):
            marker = "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(d + 0.02 if d >= 0 else d - 0.02, i, marker, va="center", fontsize=12)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel(distance_name)
    ax.set_title(title)

    if signed:
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.text(
            0.02,
            0.98,
            "Green: Higher metric \u2192 Lower accuracy (good predictor)",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color="green",
        )
        ax.text(
            0.02,
            0.93,
            "Red: Higher metric \u2192 Higher accuracy",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color="red",
        )
    else:
        max_val = max(d_values) if d_values else 1.0
        ax.set_xlim(0, max_val * 1.1)

    if any(not pd.isna(p) for p in p_values):
        y_pos = 0.88 if signed else 0.98
        ax.text(
            0.02, y_pos, "* p<0.05, ** p<0.01", transform=ax.transAxes, fontsize=9, va="top"
        )

    plt.tight_layout()
    return fig


def plot_metric_vs_accuracy(
    complexity_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    metric_name: str,
    ml_column: str = "best_accuracy",
    ax: Optional[plt.Axes] = None,
    param_column: str = "param_value",
) -> plt.Axes:
    """
    Scatter plot of a complexity metric vs ML accuracy.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        DataFrame with complexity metrics, including param_column.
    ml_df : pd.DataFrame
        DataFrame with ML performance metrics, including param_column.
    metric_name : str
        Name of the complexity metric column to plot.
    ml_column : str
        Name of the ML metric column. Default: 'best_accuracy'
    ax : plt.Axes, optional
        Axis to plot on. Creates new figure if None.
    param_column : str
        Column name for parameter values used in annotations.

    Returns
    -------
    plt.Axes
        The matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    metric_values = complexity_df[metric_name].values
    ml_values = ml_df[ml_column].values
    param_values = complexity_df[param_column].values if param_column in complexity_df.columns else None

    ax.scatter(metric_values, ml_values, s=100, alpha=0.7)

    if param_values is not None:
        for p, x, y in zip(param_values, metric_values, ml_values):
            ax.annotate(f"{p}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if len(set(metric_values)) > 1 and not np.any(np.isnan(metric_values)):
        z = np.polyfit(metric_values, ml_values, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(min(metric_values), max(metric_values), 100)
        ax.plot(x_line, p_fit(x_line), "r--", alpha=0.5)

        r, pval = stats.pearsonr(metric_values, ml_values)
        ax.set_title(f"{metric_name} vs {ml_column} (r={r:.3f}, p={pval:.3f})")
    else:
        ax.set_title(f"{metric_name} vs {ml_column}")

    ax.set_xlabel(metric_name)
    ax.set_ylabel(ml_column)

    return ax


def plot_summary(
    complexity_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    ml_column: str = "best_accuracy",
    top_n: int = 6,
    param_column: str = "param_value",
) -> plt.Figure:
    """
    Create summary plot with top correlated metrics.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        DataFrame with complexity metrics.
    ml_df : pd.DataFrame
        DataFrame with ML performance metrics.
    correlations_df : pd.DataFrame
        DataFrame with correlation results, sorted by abs_correlation.
    ml_column : str
        ML metric column to plot against. Default: 'best_accuracy'
    top_n : int
        Number of top metrics to show. Default: 6
    param_column : str
        Column name for parameter values used in annotations.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    top_metrics = correlations_df.head(top_n)["complexity_metric"].tolist()

    cols = 3
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, metric in enumerate(top_metrics):
        plot_metric_vs_accuracy(
            complexity_df, ml_df, metric, ml_column=ml_column, ax=axes[i], param_column=param_column
        )

    for j in range(len(top_metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Top Complexity Metrics Correlated with {ml_column}", y=1.02)
    plt.tight_layout()
    return fig


def plot_distance_heatmap(
    all_correlations_df: pd.DataFrame,
    ml_metric: str = "best_accuracy",
    top_n: int = 20,
) -> plt.Figure:
    """
    Create a horizontal bar plot of distances for a specific ML metric.

    Parameters
    ----------
    all_correlations_df : pd.DataFrame
        DataFrame with all distances, including 'ml_metric' column.
    ml_metric : str
        Which ML metric to filter by. Default: 'best_accuracy'
    top_n : int
        Number of top distances to show. Default: 20

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    subset = all_correlations_df[all_correlations_df["ml_metric"] == ml_metric].head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["green" if d < 0 else "red" for d in subset["distance"]]
    ax.barh(range(len(subset)), subset["distance"], color=colors, alpha=0.7)

    for i, (_, row) in enumerate(subset.iterrows()):
        if not pd.isna(row["p_value"]):
            marker = "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            d = row["distance"]
            ax.text(d + 0.02 if d >= 0 else d - 0.08, i, marker, va="center", fontsize=10)

    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(subset["complexity_metric"])
    ax.set_xlabel(f"Distance with {ml_metric}")
    ax.set_title(f"Complexity Metrics vs {ml_metric}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)

    plt.tight_layout()
    return fig


def plot_metrics_vs_parameter(
    complexity_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    param_label_col: str = "param_label",
    title: str = "Metrics vs Parameter",
    ml_prefixes: tuple = ("best_", "mean_"),
    split_subplots: bool = True,
    x_label: str = "Parameter value",
) -> plt.Figure:
    """
    Line plot of all normalised complexity and ML metrics vs. a varied parameter.

    Each metric is min-max normalised independently across parameter values so
    that all lines share the same y-axis scale.  Where ``{metric}_std`` columns
    are present in the DataFrames (produced by multi-seed experiments), error
    bars scaled by the same normalisation are drawn around each line.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        DataFrame with complexity metrics and a ``param_label`` column.
        May also contain ``{metric}_std`` columns for error bars.
    ml_df : pd.DataFrame
        DataFrame with ML performance metrics.
        May also contain ``{metric}_std`` columns for error bars.
    param_label_col : str
        Column in ``complexity_df`` used as x-axis labels. Default: 'param_label'
    title : str
        Plot title.
    ml_prefixes : tuple of str
        Only ML columns whose name starts with one of these prefixes are
        included (e.g. ``best_accuracy``, ``mean_f1``).  Per-model columns are
        excluded. Default: ('best_', 'mean_')
    split_subplots : bool
        If True (default), complexity metrics are shown in the left subplot and
        ML metrics in the right subplot.  If False, all metrics share a single
        axes (complexity = solid lines, ML = dashed lines).

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # --- collect columns ---
    exclude_cols = {"param_value", "param_label"}
    complexity_cols = [
        c for c in complexity_df.columns
        if c not in exclude_cols and not c.endswith("_std")
    ]
    ml_cols = [
        c for c in ml_df.columns
        if any(c.startswith(p) for p in ml_prefixes) and not c.endswith("_std")
    ]

    x_labels = complexity_df[param_label_col].tolist()
    n_points = len(x_labels)
    x = np.arange(n_points)

    def _normalise(values: np.ndarray) -> tuple[Optional[np.ndarray], float, float]:
        """Min-max normalise; returns (normalised, lo, hi).

        Returns (None, 0, 0) if all-NaN, (flat 0.5, lo, lo) if zero range.
        """
        if np.all(np.isnan(values)):
            return None, 0.0, 0.0
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if hi == lo:
            return np.full_like(values, 0.5, dtype=float), lo, lo
        return (values - lo) / (hi - lo), lo, hi

    def _normalise_std(std: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Scale std by the same factor used for min-max normalisation."""
        if hi == lo:
            return np.zeros_like(std, dtype=float)
        return std / (hi - lo)

    # --- build normalised series (mean + optional std) ---
    complexity_series: dict = {}
    complexity_std_series: dict = {}
    for col in complexity_cols:
        norm, lo, hi = _normalise(complexity_df[col].values.astype(float))
        if norm is not None:
            complexity_series[col] = norm
            std_col = f"{col}_std"
            if std_col in complexity_df.columns:
                complexity_std_series[col] = _normalise_std(
                    complexity_df[std_col].values.astype(float), lo, hi
                )

    ml_series: dict = {}
    ml_std_series: dict = {}
    for col in ml_cols:
        norm, lo, hi = _normalise(ml_df[col].values.astype(float))
        if norm is not None:
            ml_series[col] = norm
            std_col = f"{col}_std"
            if std_col in ml_df.columns:
                ml_std_series[col] = _normalise_std(
                    ml_df[std_col].values.astype(float), lo, hi
                )

    # --- colours ---
    cmap = plt.get_cmap("tab20")
    n_c = len(complexity_series)
    n_m = len(ml_series)
    complexity_colors = [cmap(i / max(n_c, 1)) for i in range(n_c)]
    ml_colors = [cmap(i / max(n_m, 1)) for i in range(n_m)]

    def _draw(
        ax: plt.Axes,
        series: dict,
        std_series: dict,
        colors: list,
        linestyle: str,
    ) -> None:
        for (col, values), color in zip(series.items(), colors):
            yerr = std_series.get(col)
            ax.errorbar(
                x, values,
                yerr=yerr,
                marker="o",
                linestyle=linestyle,
                color=color,
                label=col,
                linewidth=1.5,
                capsize=3,
                elinewidth=0.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Normalised value (min\u2013max per metric)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)

    if split_subplots:
        fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(20, 6))
        _draw(ax_c, complexity_series, complexity_std_series, complexity_colors, "-")
        ax_c.set_title("Complexity metrics")
        _draw(ax_m, ml_series, ml_std_series, ml_colors, "-")
        ax_m.set_title("ML metrics")
        fig.suptitle(title)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        _draw(ax, complexity_series, complexity_std_series, complexity_colors, "-")
        for (col, values), color in zip(ml_series.items(), ml_colors):
            yerr = ml_std_series.get(col)
            ax.errorbar(
                x, values,
                yerr=yerr,
                marker="o",
                linestyle="--",
                color=color,
                label=col,
                linewidth=1.5,
                capsize=3,
                elinewidth=0.8,
            )
        ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_models_vs_parameter(
    ml_df: pd.DataFrame,
    param_label_col: str = "param_label",
    title: str = "Models vs Parameter",
    ml_metrics: Optional[List[str]] = None,
    x_label: str = "Parameter value",
) -> plt.Figure:
    """
    Line plots showing each ML model in its own subplot, with one line per metric.

    Raw (un-normalised) metric values are plotted so that subplots are directly
    comparable on the same 0–1 y-axis scale.  Where ``{model}_{metric}_std``
    columns are present, error bars are drawn.

    Parameters
    ----------
    ml_df : pd.DataFrame
        DataFrame with ML performance columns named ``{model}_{metric}`` and
        optionally ``{model}_{metric}_std``.  Must also contain ``param_label``.
    param_label_col : str
        Column used as x-axis tick labels. Default: 'param_label'
    title : str
        Overall figure title.
    ml_metrics : list of str, optional
        Metric names to plot (e.g. ``['accuracy', 'f1']``).  If None, all
        metrics are inferred from the column names.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    col = param_label_col if param_label_col in ml_df.columns else "param_value"
    x_labels = ml_df[col].tolist()
    x = np.arange(len(x_labels))

    # --- infer model names and metrics from columns ---
    # Columns look like:  {model}_{metric}  and  {model}_{metric}_std
    # Aggregate columns like best_* / mean_* / param_* are excluded.
    exclude_prefixes = ("best_", "mean_", "param_")
    candidate_cols = [
        c for c in ml_df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and not c.endswith("_std")
        and c != param_label_col
    ]

    if ml_metrics is None:
        # Collect all metric suffixes that appear in the columns
        all_suffixes: set = set()
        for col in candidate_cols:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                all_suffixes.add(parts[1])
        ml_metrics = sorted(all_suffixes)

    # Build { model_name: [col_for_metric1, col_for_metric2, ...] }
    model_metric_cols: dict = {}
    for col in candidate_cols:
        for metric in ml_metrics:
            suffix = f"_{metric}"
            if col.endswith(suffix):
                model_name = col[: -len(suffix)]
                model_metric_cols.setdefault(model_name, {})[metric] = col
                break

    model_names = sorted(model_metric_cols.keys())
    n_models = len(model_names)
    if n_models == 0:
        fig, ax = plt.subplots()
        ax.set_title("No per-model columns found")
        return fig

    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    cmap = plt.get_cmap("tab10")
    metric_colors = {m: cmap(i / max(len(ml_metrics), 1)) for i, m in enumerate(ml_metrics)}

    for idx, model_name in enumerate(model_names):
        ax = flat_axes[idx]
        metric_cols = model_metric_cols[model_name]

        for metric, col in metric_cols.items():
            values = ml_df[col].values.astype(float)
            std_col = f"{col}_std"
            yerr = ml_df[std_col].values.astype(float) if std_col in ml_df.columns else None
            ax.errorbar(
                x, values,
                yerr=yerr,
                marker="o",
                linestyle="-",
                color=metric_colors[metric],
                label=metric,
                linewidth=1.5,
                capsize=3,
                elinewidth=0.8,
            )

        ax.set_title(model_name)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_xlabel(x_label)
        ax.legend(fontsize=8)

    # Turn off unused subplots
    for idx in range(n_models, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_complexity_metrics_vs_parameter(
    complexity_df: pd.DataFrame,
    param_label_col: str = "param_label",
    title: str = "Complexity Metrics vs Parameter",
    x_label: str = "Parameter value",
) -> plt.Figure:
    """
    Plot each complexity metric in its own subplot as the experiment parameter varies.

    One subplot per complexity metric, with error bars from ``{metric}_std`` columns
    where present.  Y-axes are auto-scaled per subplot since complexity metrics have
    heterogeneous ranges.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        DataFrame with complexity metric columns and optionally ``{metric}_std``
        columns.  Must contain ``param_value`` and/or ``param_label``.
    param_label_col : str
        Column used as x-axis tick labels. Default: 'param_label'
    title : str
        Overall figure title.
    x_label : str
        X-axis label applied to bottom-row subplots.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    col = param_label_col if param_label_col in complexity_df.columns else "param_value"
    x_labels = complexity_df[col].tolist()
    x = np.arange(len(x_labels))

    # Identify metric columns: exclude param_*, *_std columns
    exclude_cols = {"param_value", "param_label"}
    metric_cols = [
        c for c in complexity_df.columns
        if c not in exclude_cols and not c.endswith("_std")
    ]

    n_metrics = len(metric_cols)
    if n_metrics == 0:
        fig, ax = plt.subplots()
        ax.set_title("No complexity metric columns found")
        return fig

    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for idx, metric in enumerate(metric_cols):
        ax = flat_axes[idx]
        values = complexity_df[metric].values.astype(float)
        std_col = f"{metric}_std"
        yerr = complexity_df[std_col].values.astype(float) if std_col in complexity_df.columns else None
        ax.errorbar(
            x, values,
            yerr=yerr,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            capsize=3,
            elinewidth=0.8,
        )
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(x_label)

    # Turn off unused subplots
    for idx in range(n_metrics, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_models_vs_parameter_combined(
    train_ml_df: pd.DataFrame,
    test_ml_df: pd.DataFrame,
    param_label_col: str = "param_label",
    title: str = "Models vs Parameter (Train vs Test)",
    ml_metrics: Optional[List[str]] = None,
    x_label: str = "Parameter value",
) -> plt.Figure:
    """
    Line plots showing each ML model in its own subplot, overlaying train and test.

    Train data is plotted with solid lines; test data with dashed lines.  Both
    use the same colour per metric so train/test pairs are visually linked.

    Parameters
    ----------
    train_ml_df : pd.DataFrame
        ML performance on the training set.
    test_ml_df : pd.DataFrame
        ML performance on the test set.
    param_label_col : str
        Column used as x-axis tick labels. Default: 'param_label'
    title : str
        Overall figure title.
    ml_metrics : list of str, optional
        Metric names to plot.  Inferred from column names if None.
    x_label : str
        X-axis label.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    col = param_label_col if param_label_col in train_ml_df.columns else "param_value"
    x_labels = train_ml_df[col].tolist()
    x = np.arange(len(x_labels))

    exclude_prefixes = ("best_", "mean_", "param_")
    candidate_cols = [
        c for c in train_ml_df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and not c.endswith("_std")
        and c != param_label_col
    ]

    if ml_metrics is None:
        all_suffixes: set = set()
        for c in candidate_cols:
            parts = c.rsplit("_", 1)
            if len(parts) == 2:
                all_suffixes.add(parts[1])
        ml_metrics = sorted(all_suffixes)

    model_metric_cols: dict = {}
    for c in candidate_cols:
        for metric in ml_metrics:
            if c.endswith(f"_{metric}"):
                model_name = c[: -len(f"_{metric}")]
                model_metric_cols.setdefault(model_name, {})[metric] = c
                break

    model_names = sorted(model_metric_cols.keys())
    n_models = len(model_names)
    if n_models == 0:
        fig, ax = plt.subplots()
        ax.set_title("No per-model columns found")
        return fig

    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    cmap = plt.get_cmap("tab10")
    metric_colors = {m: cmap(i / max(len(ml_metrics), 1)) for i, m in enumerate(ml_metrics)}

    for idx, model_name in enumerate(model_names):
        ax = flat_axes[idx]
        metric_cols = model_metric_cols[model_name]

        for metric, data_col in metric_cols.items():
            color = metric_colors[metric]
            std_col = f"{data_col}_std"

            # Train — solid
            train_values = train_ml_df[data_col].values.astype(float)
            train_yerr = train_ml_df[std_col].values.astype(float) if std_col in train_ml_df.columns else None
            ax.errorbar(
                x, train_values,
                yerr=train_yerr,
                marker="o",
                linestyle="-",
                color=color,
                label=f"{metric} (train)",
                linewidth=1.5,
                capsize=3,
                elinewidth=0.8,
            )

            # Test — dashed
            if data_col in test_ml_df.columns:
                test_values = test_ml_df[data_col].values.astype(float)
                test_yerr = test_ml_df[std_col].values.astype(float) if std_col in test_ml_df.columns else None
                ax.errorbar(
                    x, test_values,
                    yerr=test_yerr,
                    marker="s",
                    linestyle="--",
                    color=color,
                    label=f"{metric} (test)",
                    linewidth=1.5,
                    capsize=3,
                    elinewidth=0.8,
                )

        ax.set_title(model_name)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_xlabel(x_label)
        ax.legend(fontsize=8)

    for idx in range(n_models, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_complexity_metrics_vs_parameter_combined(
    train_complexity_df: pd.DataFrame,
    test_complexity_df: pd.DataFrame,
    param_label_col: str = "param_label",
    title: str = "Complexity Metrics vs Parameter (Train vs Test)",
    x_label: str = "Parameter value",
) -> plt.Figure:
    """
    Plot each complexity metric in its own subplot overlaying train and test data.

    Train data is plotted with a solid line; test data with a dashed line.  Both
    share the same default colour cycle colour so the pair is visually linked.

    Parameters
    ----------
    train_complexity_df : pd.DataFrame
        Complexity metrics computed on the training set.
    test_complexity_df : pd.DataFrame
        Complexity metrics computed on the test set.
    param_label_col : str
        Column used as x-axis tick labels. Default: 'param_label'
    title : str
        Overall figure title.
    x_label : str
        X-axis label applied to bottom-row subplots.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    col = param_label_col if param_label_col in train_complexity_df.columns else "param_value"
    x_labels = train_complexity_df[col].tolist()
    x = np.arange(len(x_labels))

    exclude_cols = {"param_value", "param_label"}
    metric_cols = [
        c for c in train_complexity_df.columns
        if c not in exclude_cols and not c.endswith("_std")
    ]

    n_metrics = len(metric_cols)
    if n_metrics == 0:
        fig, ax = plt.subplots()
        ax.set_title("No complexity metric columns found")
        return fig

    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    cmap = plt.get_cmap("tab10")

    for idx, metric in enumerate(metric_cols):
        ax = flat_axes[idx]
        color = cmap(0)  # single colour per subplot; train/test distinguished by linestyle

        # Train — solid
        train_values = train_complexity_df[metric].values.astype(float)
        std_col = f"{metric}_std"
        train_yerr = (
            train_complexity_df[std_col].values.astype(float)
            if std_col in train_complexity_df.columns
            else None
        )
        ax.errorbar(
            x, train_values,
            yerr=train_yerr,
            marker="o",
            linestyle="-",
            color=color,
            label="train",
            linewidth=1.5,
            capsize=3,
            elinewidth=0.8,
        )

        # Test — dashed
        if metric in test_complexity_df.columns:
            test_values = test_complexity_df[metric].values.astype(float)
            test_yerr = (
                test_complexity_df[std_col].values.astype(float)
                if std_col in test_complexity_df.columns
                else None
            )
            ax.errorbar(
                x, test_values,
                yerr=test_yerr,
                marker="s",
                linestyle="--",
                color=color,
                label="test",
                linewidth=1.5,
                capsize=3,
                elinewidth=0.8,
            )

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(x_label)
        ax.legend(fontsize=8)

    for idx in range(n_metrics, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_datasets_overview(
    datasets: Dict[Any, Any],
    format_label: Callable[[Any], str],
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> plt.Figure:
    """
    Composite grid showing all dataset visualizations.

    Rows = one per parameter value; columns = Full Dataset | Train | Test.
    All subplots share the same axis scale limits. X-axis labels appear only
    on the bottom row. Parameter values are shown as row labels in the left
    margin, outside the plot area, preserving the original y-axis labels.

    Parameters
    ----------
    datasets : dict
        param_value -> dataset object (from Experiment.datasets).
    format_label : callable
        Converts a param_value to a human-readable row label.
    cell_width : float
        Width of each grid cell in inches. Default: 6.0
    cell_height : float
        Height of each grid cell in inches. Default: 4.0

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    param_values = list(datasets.keys())
    n_rows = len(param_values)

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(cell_width * 3, cell_height * n_rows),
        squeeze=False,
    )

    col_titles = ["Full Dataset", "Train", "Test"]
    for col_idx, title in enumerate(col_titles):
        axes[0][col_idx].set_title(title, fontsize=12, fontweight="bold")

    for row_idx, param_value in enumerate(param_values):
        dataset = datasets[param_value]
        row_axes = axes[row_idx]
        try:
            dataset.plot_dataset(ax=row_axes[0])
            dataset.plot_train_test_split(ax=(row_axes[1], row_axes[2]))
        except (ValueError, AttributeError):
            dataset.plot_dataset(ax=row_axes[0])
            for blank_ax in row_axes[1:]:
                blank_ax.axis("off")
                blank_ax.text(
                    0.5, 0.5, "N/A",
                    transform=blank_ax.transAxes,
                    ha="center", va="center", fontsize=12, color="grey",
                )
        # Add param label to the left margin, outside the axes area
        row_axes[0].text(
            -0.18, 0.5,
            format_label(param_value),
            transform=row_axes[0].transAxes,
            fontsize=14,
            ha="center",
            va="center",
            rotation=90,
            clip_on=False,
            fontweight="bold"
        )

    # Uniform axis limits across all populated axes
    all_xlims = [ax.get_xlim() for row in axes for ax in row if ax.has_data()]
    all_ylims = [ax.get_ylim() for row in axes for ax in row if ax.has_data()]
    if all_xlims:
        global_xlim = (min(lo for lo, _ in all_xlims), max(hi for _, hi in all_xlims))
        global_ylim = (min(lo for lo, _ in all_ylims), max(hi for _, hi in all_ylims))
        for row in axes:
            for ax in row:
                if ax.has_data():
                    ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

    # X-axis labels only on the bottom row and title only on the top row; row labels in the left margin
    for row_idx, row in enumerate(axes):
        if row_idx < n_rows - 1:
            for ax in row:
                ax.set_xlabel("")
        if row_idx > 0:
            for ax in row:
                ax.set_title("")

    fig.suptitle("Datasets Overview", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(left=0.1)
    return fig


def plot_pairwise_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Pairwise Distances",
) -> plt.Figure:
    """
    Plot a heatmap of a pairwise distance/association matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        N×N symmetric matrix (from DataFrame.corr() or similar).
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    n = len(corr_matrix)
    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    values = corr_matrix.values.astype(float)
    masked_values = np.ma.masked_invalid(values)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(masked_values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Distance")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_matrix.index, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            if np.isnan(val):
                ax.text(j, i, "\u2014", ha="center", va="center", fontsize=6, color="gray")
            else:
                text_color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_model_comparison(
    all_correlations_df: pd.DataFrame,
    complexity_metrics: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot correlation of complexity metrics with each model's accuracy.

    Parameters
    ----------
    all_correlations_df : pd.DataFrame
        DataFrame with all correlations.
    complexity_metrics : list of str, optional
        Complexity metrics to include. If None, uses top 5 by best_accuracy.
    model_names : list of str, optional
        Model names to include. If None, extracts from ml_metric column.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    if complexity_metrics is None:
        top = all_correlations_df[all_correlations_df["ml_metric"] == "best_accuracy"].head(5)
        complexity_metrics = top["complexity_metric"].tolist()

    if model_names is None:
        ml_cols = all_correlations_df["ml_metric"].unique()
        model_names = [c.replace("_accuracy", "") for c in ml_cols if c.endswith("_accuracy") and not c.startswith("best") and not c.startswith("mean")]

    matrix = np.zeros((len(complexity_metrics), len(model_names)))
    for i, cm in enumerate(complexity_metrics):
        for j, model in enumerate(model_names):
            ml_col = f"{model}_accuracy"
            subset = all_correlations_df[
                (all_correlations_df["complexity_metric"] == cm)
                & (all_correlations_df["ml_metric"] == ml_col)
            ]
            if len(subset) > 0:
                matrix[i, j] = subset.iloc[0]["distance"]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticks(range(len(complexity_metrics)))
    ax.set_yticklabels(complexity_metrics)

    for i in range(len(complexity_metrics)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, label="Correlation")
    ax.set_title("Complexity Metric Correlations with Each Model's Accuracy")
    plt.tight_layout()
    return fig
