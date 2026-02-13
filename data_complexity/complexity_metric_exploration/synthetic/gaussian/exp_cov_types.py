"""
Experiment: Effect of covariance structure on complexity metrics.

Compares different covariance types: spherical, diagonal, symmetric, random.
Different covariance structures affect class boundaries and overlap patterns.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import ComplexityMetrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(cov_types=None, plot_datasets=False, terminal_plot=True):
    """
    Run covariance type comparison experiment.

    Parameters
    ----------
    cov_types : list of str
        Covariance types to test. Default: ['spherical', 'diagonal', 'symmetric', 'random']
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if cov_types is None:
        cov_types = ['spherical', 'diagonal', 'symmetric', 'random']

    metrics_by_dataset = {}

    for cov_type in cov_types:
        dataset_name = cov_type
        dataset = get_dataset(
            "Gaussian",
            class_separation=3.0,
            cov_type=cov_type,
            cov_scale=1.0,
            cov_correlation=0.7,  # Only used for 'symmetric'
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Covariance Type Effect on Complexity")
    print("=" * 50)
    print("Comparing: spherical, diagonal, symmetric, random")
    print()

    metrics = run_experiment(
        cov_types=['spherical', 'diagonal', 'symmetric', 'random'],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
