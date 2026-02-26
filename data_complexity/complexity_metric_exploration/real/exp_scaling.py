"""
Experiment: Effect of feature scaling on complexity metrics.

Tests whether standardizing features affects complexity measurements.
Some metrics may be sensitive to feature scales.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.data_metrics.metrics import ComplexityMetrics
from data_complexity.plotting.plot_multiple_datasets import plot_metrics


def run_experiment(dataset_name="Wine", plot_datasets=False, terminal_plot=True):
    """
    Run scaling effect experiment on a single dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset to test. Default: "Wine"
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by scaling condition.
    """
    metrics_by_dataset = {}

    for scaled, label in [(False, "unscaled"), (True, "scaled")]:
        dataset = get_dataset(dataset_name, scale=scaled)

        if plot_datasets:
            print(f"\n{label}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[label] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Feature Scaling Effect on Complexity")
    print("=" * 50)
    print("Comparing scaled vs unscaled features")
    print()

    metrics = run_experiment(
        dataset_name="Wine",
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
