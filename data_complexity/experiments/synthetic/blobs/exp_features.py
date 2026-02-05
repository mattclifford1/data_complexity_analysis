"""
Experiment: Effect of dimensionality on blobs dataset complexity.

Tests how the number of features affects complexity metrics.
Higher dimensions may affect distance-based metrics.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import complexity_metrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(feature_counts=None, plot_datasets=False, terminal_plot=True):
    """
    Run blobs dimensionality experiment.

    Parameters
    ----------
    feature_counts : list of int
        Number of features to test. Default: [2, 5, 10, 20]
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if feature_counts is None:
        feature_counts = [2, 5, 10, 20]

    metrics_by_dataset = {}

    for n_features in feature_counts:
        dataset_name = f'd={n_features}'
        dataset = get_dataset(
            "Blobs",
            blobs_features=n_features,
            num_samples=400,
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = complexity_metrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Blobs Dimensionality Effect on Complexity")
    print("=" * 50)
    print("Testing curse of dimensionality effects")
    print()

    metrics = run_experiment(
        feature_counts=[2, 5, 10, 20],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
