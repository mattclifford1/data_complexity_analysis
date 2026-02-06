"""
Plotting utilities for experiment visualization.

Provides reusable plotting functions for complexity vs ML performance analysis.
"""
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_correlations(
    correlations_df: pd.DataFrame,
    title: str = "Complexity vs ML Accuracy Correlations",
    top_n: int = 15,
) -> plt.Figure:
    """
    Plot correlation coefficients as a horizontal bar chart.

    Parameters
    ----------
    correlations_df : pd.DataFrame
        DataFrame with columns: 'complexity_metric', 'correlation', 'p_value'.
        Must be sorted by abs_correlation descending.
    title : str
        Plot title.
    top_n : int
        Number of top correlations to display. Default: 15

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    subset = correlations_df.head(top_n)

    metrics = subset["complexity_metric"].tolist()
    r_values = subset["correlation"].tolist()
    p_values = subset["p_value"].tolist()

    colors = ["green" if r < 0 else "red" for r in r_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(metrics)), r_values, color=colors, alpha=0.7)

    for i, (r, p) in enumerate(zip(r_values, p_values)):
        marker = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(r + 0.02 if r >= 0 else r - 0.02, i, marker, va="center", fontsize=12)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title(title)
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
    ax.text(
        0.02, 0.88, "* p<0.05, ** p<0.01", transform=ax.transAxes, fontsize=9, va="top"
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


def plot_correlation_heatmap(
    all_correlations_df: pd.DataFrame,
    ml_metric: str = "best_accuracy",
    top_n: int = 20,
) -> plt.Figure:
    """
    Create a horizontal bar plot of correlations for a specific ML metric.

    Parameters
    ----------
    all_correlations_df : pd.DataFrame
        DataFrame with all correlations, including 'ml_metric' column.
    ml_metric : str
        Which ML metric to filter by. Default: 'best_accuracy'
    top_n : int
        Number of top correlations to show. Default: 20

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    subset = all_correlations_df[all_correlations_df["ml_metric"] == ml_metric].head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["green" if r < 0 else "red" for r in subset["correlation"]]
    ax.barh(range(len(subset)), subset["correlation"], color=colors, alpha=0.7)

    for i, (_, row) in enumerate(subset.iterrows()):
        marker = "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        r = row["correlation"]
        ax.text(r + 0.02 if r >= 0 else r - 0.08, i, marker, va="center", fontsize=10)

    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(subset["complexity_metric"])
    ax.set_xlabel(f"Pearson Correlation with {ml_metric}")
    ax.set_title(f"Complexity Metrics vs {ml_metric}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)

    plt.tight_layout()
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
                matrix[i, j] = subset.iloc[0]["correlation"]

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
