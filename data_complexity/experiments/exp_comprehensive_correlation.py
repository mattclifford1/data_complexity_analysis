"""
Experiment: Comprehensive correlation analysis across multiple dataset variations.

Combines multiple experiments to find which complexity metrics are
the most consistent predictors of ML performance across multiple
models and evaluation metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import complexity_metrics
from data_complexity.experiments.ml_evaluation import (
    evaluate_classifiers,
    get_default_classifiers,
    get_metric_summary,
    get_best_metric,
    get_mean_metric,
    SCORING_METRICS,
)

# Output directory for results
RESULTS_DIR = Path(__file__).parent / "results" / "comprehensive"


def collect_all_datasets():
    """
    Generate a diverse collection of datasets with varying parameters.

    Returns
    -------
    list of dict
        Each dict has 'name', 'X', 'y' keys.
    """
    datasets = []

    def add_dataset(name, ds):
        data = ds.get_data_dict()
        datasets.append({"name": name, "X": data["X"], "y": data["y"]})

    # Gaussian: vary separation
    for sep in [1.0, 2.0, 4.0, 6.0, 8.0]:
        ds = get_dataset("Gaussian", class_separation=sep, cov_scale=1.0, num_samples=400)
        add_dataset(f"Gauss_sep{sep}", ds)

    # Gaussian: vary scale
    for scale in [0.5, 1.0, 2.0, 4.0]:
        ds = get_dataset("Gaussian", class_separation=4.0, cov_scale=scale, num_samples=400)
        add_dataset(f"Gauss_scale{scale}", ds)

    # Gaussian: vary correlation
    for corr in [-0.5, 0.0, 0.5]:
        ds = get_dataset("Gaussian", class_separation=4.0, cov_type="symmetric",
                         cov_correlation=corr, num_samples=400)
        add_dataset(f"Gauss_corr{corr}", ds)

    # Moons: vary noise
    for noise in [0.05, 0.1, 0.2, 0.4]:
        ds = get_dataset("Moons", moons_noise=noise, num_samples=400)
        add_dataset(f"Moons_noise{noise}", ds)

    # Circles: vary noise
    for noise in [0.02, 0.05, 0.1, 0.2]:
        ds = get_dataset("Circles", circles_noise=noise, num_samples=400)
        add_dataset(f"Circles_noise{noise}", ds)

    # Blobs: vary features
    for n_feat in [2, 5, 10]:
        ds = get_dataset("Blobs", blobs_features=n_feat, num_samples=400)
        add_dataset(f"Blobs_d{n_feat}", ds)

    # XOR
    ds = get_dataset("XOR", num_samples=[200, 200])
    add_dataset("XOR", ds)

    return datasets


def run_experiment(datasets=None, cv_folds=5):
    """
    Run comprehensive complexity vs ML experiment.

    Parameters
    ----------
    datasets : list of dict, optional
        Datasets to evaluate. Default: collect_all_datasets()
    cv_folds : int
        Cross-validation folds. Default: 5

    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame)
        - complexity_df: Complexity metrics for each dataset
        - ml_df: ML performance (all models, all metrics) for each dataset
    """
    if datasets is None:
        datasets = collect_all_datasets()

    complexity_rows = []
    ml_rows = []
    model_names = list(get_default_classifiers().keys())

    for ds in datasets:
        print(f"Processing {ds['name']}...")

        X, y = ds["X"], ds["y"]

        # Compute complexity metrics
        complexity = complexity_metrics(dataset={"X": X, "y": y})
        metrics = complexity.get_all_metrics_scalar()
        complexity_row = {"dataset": ds["name"]}
        complexity_row.update(metrics)
        complexity_rows.append(complexity_row)

        # Evaluate ML models
        ml_results = evaluate_classifiers(X, y, cv_folds=cv_folds)

        ml_row = {"dataset": ds["name"]}

        # Add best/mean for each metric
        for metric in SCORING_METRICS.keys():
            ml_row[f"best_{metric}"] = get_best_metric(ml_results, metric)
            ml_row[f"mean_{metric}"] = get_mean_metric(ml_results, metric)

        # Add per-model results
        for model in model_names:
            for metric in SCORING_METRICS.keys():
                if model in ml_results and metric in ml_results[model]:
                    ml_row[f"{model}_{metric}"] = ml_results[model][metric]["mean"]
                else:
                    ml_row[f"{model}_{metric}"] = np.nan

        ml_rows.append(ml_row)

    return pd.DataFrame(complexity_rows), pd.DataFrame(ml_rows)


def compute_correlations(complexity_df, ml_df, ml_column="best_accuracy"):
    """
    Compute correlations between complexity metrics and an ML performance column.

    Parameters
    ----------
    complexity_df : pd.DataFrame
        Complexity metrics.
    ml_df : pd.DataFrame
        ML performance metrics.
    ml_column : str
        Column from ml_df to correlate against.

    Returns
    -------
    pd.DataFrame
        Correlation results sorted by absolute correlation.
    """
    metric_cols = [c for c in complexity_df.columns if c != "dataset"]
    ml_values = ml_df[ml_column].values
    results = []

    for metric in metric_cols:
        values = complexity_df[metric].values

        # Skip if constant or has NaN
        if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
            continue

        r, p = stats.pearsonr(values, ml_values)
        results.append({
            "complexity_metric": metric,
            "ml_metric": ml_column,
            "correlation": r,
            "p_value": p,
            "abs_correlation": abs(r),
        })

    return pd.DataFrame(results).sort_values("abs_correlation", ascending=False)


def compute_all_correlations(complexity_df, ml_df):
    """
    Compute correlations for all ML metrics and models.

    Returns
    -------
    pd.DataFrame
        All correlations.
    """
    all_corrs = []
    ml_cols = [c for c in ml_df.columns if c != "dataset"]

    for ml_col in ml_cols:
        corr_df = compute_correlations(complexity_df, ml_df, ml_col)
        all_corrs.append(corr_df)

    return pd.concat(all_corrs, ignore_index=True)


def print_model_performance(ml_df):
    """Print average performance across all datasets for each model."""
    model_names = list(get_default_classifiers().keys())

    print("\n" + "=" * 80)
    print("ML MODEL PERFORMANCE (averaged across all datasets)")
    print("=" * 80)

    for metric in SCORING_METRICS.keys():
        print(f"\n{metric.upper()}:")
        print("-" * 50)
        print(f"{'Model':<20} {'Mean':>10} {'Std':>10}")
        print("-" * 50)

        model_scores = []
        for model in model_names:
            col = f"{model}_{metric}"
            if col in ml_df.columns:
                mean_val = ml_df[col].mean()
                std_val = ml_df[col].std()
                model_scores.append((model, mean_val, std_val))

        # Sort by mean score
        model_scores.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else float('-inf'))
        for model, mean_val, std_val in model_scores:
            print(f"{model:<20} {mean_val:>10.4f} {std_val:>10.4f}")


def print_top_correlations(all_corr_df, top_n=10):
    """Print top correlations for each ML metric type."""
    print("\n" + "=" * 80)
    print("TOP COMPLEXITY-PERFORMANCE CORRELATIONS")
    print("=" * 80)

    # Group by ML metric type (best_accuracy, mean_accuracy, etc.)
    ml_metric_types = ["best_accuracy", "best_f1", "mean_accuracy", "mean_f1"]

    for ml_type in ml_metric_types:
        subset = all_corr_df[all_corr_df["ml_metric"] == ml_type].head(top_n)
        if len(subset) == 0:
            continue

        print(f"\nTop {top_n} correlations with {ml_type}:")
        print("-" * 60)
        print(f"{'Complexity Metric':<25} {'Correlation':>12} {'p-value':>12}")
        print("-" * 60)

        for _, row in subset.iterrows():
            sig = "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"{row['complexity_metric']:<25} {row['correlation']:>+12.3f} {row['p_value']:>12.4f} {sig}")


def print_per_model_correlations(all_corr_df, complexity_metric="N3"):
    """Print correlations for a specific complexity metric across all models."""
    print(f"\n" + "=" * 80)
    print(f"CORRELATIONS OF '{complexity_metric}' WITH EACH MODEL'S ACCURACY")
    print("=" * 80)

    model_names = list(get_default_classifiers().keys())

    print(f"\n{'Model':<20} {'Correlation':>12} {'p-value':>12}")
    print("-" * 50)

    for model in model_names:
        ml_col = f"{model}_accuracy"
        subset = all_corr_df[
            (all_corr_df["complexity_metric"] == complexity_metric) &
            (all_corr_df["ml_metric"] == ml_col)
        ]
        if len(subset) > 0:
            row = subset.iloc[0]
            sig = "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"{model:<20} {row['correlation']:>+12.3f} {row['p_value']:>12.4f} {sig}")


def plot_correlation_heatmap(all_corr_df, ml_metric="best_accuracy", top_n=20):
    """Create a horizontal bar plot of correlations."""
    subset = all_corr_df[all_corr_df["ml_metric"] == ml_metric].head(top_n)

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


def plot_model_comparison(all_corr_df, complexity_metrics=None):
    """
    Plot correlation of top complexity metrics with each model's accuracy.
    """
    if complexity_metrics is None:
        # Get top 5 complexity metrics by best_accuracy correlation
        top = all_corr_df[all_corr_df["ml_metric"] == "best_accuracy"].head(5)
        complexity_metrics = top["complexity_metric"].tolist()

    model_names = list(get_default_classifiers().keys())

    # Build matrix
    matrix = np.zeros((len(complexity_metrics), len(model_names)))
    for i, cm in enumerate(complexity_metrics):
        for j, model in enumerate(model_names):
            ml_col = f"{model}_accuracy"
            subset = all_corr_df[
                (all_corr_df["complexity_metric"] == cm) &
                (all_corr_df["ml_metric"] == ml_col)
            ]
            if len(subset) > 0:
                matrix[i, j] = subset.iloc[0]["correlation"]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticks(range(len(complexity_metrics)))
    ax.set_yticklabels(complexity_metrics)

    # Add correlation values
    for i in range(len(complexity_metrics)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, label="Correlation")
    ax.set_title("Complexity Metric Correlations with Each Model's Accuracy")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE COMPLEXITY vs ML PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    print("DATASET VARIATIONS:")
    print("-" * 40)
    print("Gaussian:")
    print("  - class_separation: 1.0, 2.0, 4.0, 6.0, 8.0")
    print("  - cov_scale (variance): 0.5, 1.0, 2.0, 4.0")
    print("  - cov_correlation: -0.5, 0.0, 0.5")
    print("Moons:")
    print("  - noise: 0.05, 0.1, 0.2, 0.4")
    print("Circles:")
    print("  - noise: 0.02, 0.05, 0.1, 0.2")
    print("Blobs:")
    print("  - n_features: 2, 5, 10")
    print("XOR:")
    print("  - fixed configuration")
    print()

    print("ML MODELS:")
    print("-" * 40)
    for name in get_default_classifiers().keys():
        print(f"  - {name}")
    print()

    print("ML METRICS:")
    print("-" * 40)
    for name in SCORING_METRICS.keys():
        print(f"  - {name}")
    print()

    # Run experiment
    complexity_df, ml_df = run_experiment()

    # Print model performance summary
    print_model_performance(ml_df)

    # Compute all correlations
    print("\n" + "=" * 80)
    print("Computing correlations...")
    all_corr_df = compute_all_correlations(complexity_df, ml_df)

    # Print top correlations
    print_top_correlations(all_corr_df)

    # Print per-model correlations for top complexity metric
    top_metric = all_corr_df[all_corr_df["ml_metric"] == "best_accuracy"].iloc[0]["complexity_metric"]
    print_per_model_correlations(all_corr_df, top_metric)

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {RESULTS_DIR}")

    # Save data
    complexity_df.to_csv(RESULTS_DIR / "complexity_metrics.csv", index=False)
    ml_df.to_csv(RESULTS_DIR / "ml_performance.csv", index=False)
    all_corr_df.to_csv(RESULTS_DIR / "correlations.csv", index=False)
    print("Saved CSVs:")
    print(f"  - {RESULTS_DIR / 'complexity_metrics.csv'}")
    print(f"  - {RESULTS_DIR / 'ml_performance.csv'}")
    print(f"  - {RESULTS_DIR / 'correlations.csv'}")

    # Generate plots (save only, no display)
    print("\nGenerating plots...")

    plot_correlation_heatmap(all_corr_df, "best_accuracy", top_n=20)
    plt.savefig(RESULTS_DIR / "correlations_best_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_correlation_heatmap(all_corr_df, "best_f1", top_n=20)
    plt.savefig(RESULTS_DIR / "correlations_best_f1.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_model_comparison(all_corr_df)
    plt.savefig(RESULTS_DIR / "model_comparison_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plots:")
    print(f"  - {RESULTS_DIR / 'correlations_best_accuracy.png'}")
    print(f"  - {RESULTS_DIR / 'correlations_best_f1.png'}")
    print(f"  - {RESULTS_DIR / 'model_comparison_heatmap.png'}")

    print("\nDone.")
