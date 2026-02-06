"""
Experiment: Correlate complexity with ML performance on moons dataset.

Varies the noise level in moons dataset while measuring both
complexity metrics and ML classifier accuracies.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import data_loaders

# Output directory for results
RESULTS_DIR = Path(__file__).parent / "results" / "moons_noise"
from data_loaders import get_dataset
from data_complexity.metrics import complexity_metrics
from data_complexity.experiments.ml_evaluation import (
    evaluate_classifiers,
    get_best_accuracy,
    get_mean_accuracy,
)
from data_complexity.experiments.exp_complexity_vs_ml import (
    compute_correlations,
    plot_correlations,
    plot_summary,
)


def run_experiment(
    noise_levels=None,
    num_samples=400,
    cv_folds=5,
    plot_datasets=False,
    terminal_plot=True,
):
    """
    Run moons noise vs ML performance experiment.

    Parameters
    ----------
    noise_levels : list of float
        Noise levels to test. Default: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
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
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    results = {
        "scales": noise_levels,  # Using 'scales' key for compatibility
        "complexity": {},
        "ml_performance": {},
    }

    for noise in noise_levels:
        dataset_name = f"noise={noise}"
        dataset = get_dataset(
            "Moons",
            moons_noise=noise,
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
        results["complexity"][noise] = complexity.get_all_metrics_scalar()

        # Evaluate ML models
        ml_results = evaluate_classifiers(X, y, cv_folds=cv_folds)
        results["ml_performance"][noise] = {
            "classifiers": ml_results,
            "best_accuracy": get_best_accuracy(ml_results),
            "mean_accuracy": get_mean_accuracy(ml_results),
        }

        print(f"{dataset_name}: best_acc={results['ml_performance'][noise]['best_accuracy']:.3f}")

    return results


if __name__ == "__main__":
    print("Experiment: Moons Noise vs ML Performance")
    print("=" * 50)
    print("Varying noise level in moons dataset")
    print()

    results = run_experiment(
        noise_levels=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        num_samples=400,
        plot_datasets=False,
    )

    print("\n" + "=" * 50)
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

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {RESULTS_DIR}")

    # Create subfolders for organized results
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\nGenerating plots...")
    plot_correlations(correlations, title="Moons Noise: Complexity vs Accuracy")
    plt.savefig(plots_dir / "correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'correlations.png'}")

    plot_summary(results, top_n=6)
    plt.savefig(plots_dir / "top_metrics_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'top_metrics_vs_accuracy.png'}")

    print("\nDone.")
