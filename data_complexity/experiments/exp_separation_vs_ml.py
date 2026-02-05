"""
Experiment: Correlate complexity with ML performance across class separations.

Varies the distance between Gaussian class means while measuring both
complexity metrics and ML classifier accuracies.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import data_loaders
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
    separations=None,
    cov_scale=1.0,
    num_samples=400,
    cv_folds=5,
    plot_datasets=False,
    terminal_plot=True,
):
    """
    Run class separation vs ML performance experiment.

    Parameters
    ----------
    separations : list of float
        Class separations to test. Default: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    cov_scale : float
        Covariance scale (fixed). Default: 1.0
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
    if separations is None:
        separations = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

    results = {
        "scales": separations,  # Using 'scales' key for compatibility
        "complexity": {},
        "ml_performance": {},
    }

    for sep in separations:
        dataset_name = f"sep={sep}"
        dataset = get_dataset(
            "Gaussian",
            class_separation=sep,
            cov_type="spherical",
            cov_scale=cov_scale,
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
        results["complexity"][sep] = complexity.get_all_metrics_scalar()

        # Evaluate ML models
        ml_results = evaluate_classifiers(X, y, cv_folds=cv_folds)
        results["ml_performance"][sep] = {
            "classifiers": ml_results,
            "best_accuracy": get_best_accuracy(ml_results),
            "mean_accuracy": get_mean_accuracy(ml_results),
        }

        print(f"{dataset_name}: best_acc={results['ml_performance'][sep]['best_accuracy']:.3f}")

    return results


if __name__ == "__main__":
    print("Experiment: Class Separation vs ML Performance")
    print("=" * 50)
    print("Varying distance between Gaussian class means")
    print()

    results = run_experiment(
        separations=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        cov_scale=1.0,
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

    print("\nGenerating plots...")
    plot_correlations(correlations, title="Class Separation: Complexity vs Accuracy")
    plt.savefig("separation_correlations.png", dpi=150, bbox_inches="tight")
    print("Saved: separation_correlations.png")

    plot_summary(results, top_n=6)
    plt.savefig("separation_vs_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved: separation_vs_accuracy.png")

    plt.show()
