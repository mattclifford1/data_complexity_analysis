"""
Experiment: Correlate complexity metrics with ML model performance.

Varies Gaussian covariance scale (variance) and measures both:
1. Data complexity metrics
2. ML classifier accuracies

Then computes and visualizes correlations between them.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import complexity_metrics
from data_complexity.experiments.ml_evaluation import (
    evaluate_classifiers,
    get_best_accuracy,
    get_mean_accuracy,
    get_linear_accuracy,
)

# Output directory for results
RESULTS_DIR = Path(__file__).parent / "results" / "gaussian_variance"


def run_experiment(
    scales=None,
    class_separation=4.0,
    num_samples=400,
    cv_folds=5,
    plot_datasets=False,
    terminal_plot=True,
):
    """
    Run complexity vs ML performance experiment.

    Parameters
    ----------
    scales : list of float
        Covariance scales to test. Default: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    class_separation : float
        Distance between class means. Default: 4.0
    num_samples : int
        Number of samples per dataset. Default: 400
    cv_folds : int
        Cross-validation folds. Default: 5
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Results with 'complexity', 'ml_performance', and 'scales' keys.
    """
    if scales is None:
        scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    results = {
        "scales": scales,
        "complexity": {},
        "ml_performance": {},
    }

    for scale in scales:
        dataset_name = f"scale={scale}"
        dataset = get_dataset(
            "Gaussian",
            class_separation=class_separation,
            cov_type="spherical",
            cov_scale=scale,
            num_samples=num_samples,
            name=dataset_name,
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        data = dataset.get_data_dict()
        X, y = data["X"], data["y"]

        # Compute complexity metrics
        complexity = complexity_metrics(dataset=data)
        results["complexity"][scale] = complexity.get_all_metrics_scalar()

        # Evaluate ML models
        ml_results = evaluate_classifiers(X, y, cv_folds=cv_folds)
        results["ml_performance"][scale] = {
            "classifiers": ml_results,
            "best_accuracy": get_best_accuracy(ml_results),
            "mean_accuracy": get_mean_accuracy(ml_results),
            "linear_accuracy": get_linear_accuracy(ml_results),
        }

        print(f"{dataset_name}: best_acc={results['ml_performance'][scale]['best_accuracy']:.3f}")

    return results


def compute_correlations(results):
    """
    Compute correlations between complexity metrics and ML performance.

    Parameters
    ----------
    results : dict
        Output from run_experiment().

    Returns
    -------
    dict
        Metric name -> correlation with best_accuracy (Pearson r, p-value).
    """
    scales = results["scales"]
    best_accs = [results["ml_performance"][s]["best_accuracy"] for s in scales]

    correlations = {}
    metric_names = list(results["complexity"][scales[0]].keys())

    for metric in metric_names:
        metric_values = [results["complexity"][s][metric] for s in scales]

        # Skip if constant or has NaN
        if np.std(metric_values) == 0 or np.any(np.isnan(metric_values)):
            correlations[metric] = {"r": np.nan, "p": np.nan}
            continue

        r, p = stats.pearsonr(metric_values, best_accs)
        correlations[metric] = {"r": r, "p": p}

    return correlations


def plot_correlations(correlations, title="Complexity vs ML Accuracy Correlations"):
    """
    Plot correlation coefficients as a bar chart.

    Parameters
    ----------
    correlations : dict
        Output from compute_correlations().
    title : str
        Plot title.
    """
    # Filter out NaN correlations and sort by absolute correlation
    valid = {k: v for k, v in correlations.items() if not np.isnan(v["r"])}
    sorted_metrics = sorted(valid.keys(), key=lambda x: abs(valid[x]["r"]), reverse=True)

    metrics = sorted_metrics[:15]  # Top 15
    r_values = [valid[m]["r"] for m in metrics]
    p_values = [valid[m]["p"] for m in metrics]

    # Color by sign: negative correlation (good predictor) vs positive
    colors = ["green" if r < 0 else "red" for r in r_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(metrics)), r_values, color=colors, alpha=0.7)

    # Mark significant correlations
    for i, (r, p) in enumerate(zip(r_values, p_values)):
        marker = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(r + 0.02 if r >= 0 else r - 0.02, i, marker, va="center", fontsize=12)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Pearson Correlation with Best Accuracy")
    ax.set_title(title)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)

    # Legend
    ax.text(0.02, 0.98, "Green: Higher metric → Lower accuracy (good predictor)",
            transform=ax.transAxes, fontsize=9, va="top", color="green")
    ax.text(0.02, 0.93, "Red: Higher metric → Higher accuracy",
            transform=ax.transAxes, fontsize=9, va="top", color="red")
    ax.text(0.02, 0.88, "* p<0.05, ** p<0.01",
            transform=ax.transAxes, fontsize=9, va="top")

    plt.tight_layout()
    return fig


def plot_metric_vs_accuracy(results, metric_name, ax=None):
    """
    Scatter plot of a specific metric vs ML accuracy.

    Parameters
    ----------
    results : dict
        Output from run_experiment().
    metric_name : str
        Name of complexity metric to plot.
    ax : matplotlib axis, optional
        Axis to plot on.
    """
    scales = results["scales"]
    metric_values = [results["complexity"][s][metric_name] for s in scales]
    best_accs = [results["ml_performance"][s]["best_accuracy"] for s in scales]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(metric_values, best_accs, s=100, alpha=0.7)

    # Add scale labels
    for s, x, y in zip(scales, metric_values, best_accs):
        ax.annotate(f"s={s}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Fit line
    if len(set(metric_values)) > 1:
        z = np.polyfit(metric_values, best_accs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(metric_values), max(metric_values), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5)

        r, pval = stats.pearsonr(metric_values, best_accs)
        ax.set_title(f"{metric_name} vs Accuracy (r={r:.3f}, p={pval:.3f})")
    else:
        ax.set_title(f"{metric_name} vs Accuracy")

    ax.set_xlabel(metric_name)
    ax.set_ylabel("Best Accuracy")

    return ax


def plot_summary(results, top_n=6):
    """
    Create summary plot with top correlated metrics.

    Parameters
    ----------
    results : dict
        Output from run_experiment().
    top_n : int
        Number of top metrics to show. Default: 6
    """
    correlations = compute_correlations(results)

    # Get top correlated metrics (by absolute value)
    valid = {k: v for k, v in correlations.items() if not np.isnan(v["r"])}
    top_metrics = sorted(valid.keys(), key=lambda x: abs(valid[x]["r"]), reverse=True)[:top_n]

    cols = 3
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, metric in enumerate(top_metrics):
        plot_metric_vs_accuracy(results, metric, ax=axes[i])

    # Hide unused axes
    for j in range(len(top_metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top Complexity Metrics Correlated with ML Accuracy", y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Experiment: Complexity Metrics vs ML Performance")
    print("=" * 55)
    print("Varying Gaussian covariance scale (variance)")
    print()

    results = run_experiment(
        scales=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        class_separation=4.0,
        num_samples=400,
        plot_datasets=False,
        terminal_plot=True,
    )

    print("\n" + "=" * 55)
    print("Computing correlations...")
    correlations = compute_correlations(results)

    # Print top correlations
    valid = {k: v for k, v in correlations.items() if not np.isnan(v["r"])}
    sorted_corr = sorted(valid.items(), key=lambda x: abs(x[1]["r"]), reverse=True)

    print("\nTop 10 correlations with best accuracy:")
    print("-" * 45)
    for metric, vals in sorted_corr[:10]:
        sig = "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
        print(f"  {metric:20s}: r={vals['r']:+.3f} (p={vals['p']:.3f}) {sig}")

    print("\nGenerating plots...")
    plot_correlations(correlations)
    plt.savefig("complexity_correlations.png", dpi=150, bbox_inches="tight")
    print("Saved: complexity_correlations.png")

    plot_summary(results, top_n=6)
    plt.savefig("complexity_vs_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved: complexity_vs_accuracy.png")

    plt.show()
