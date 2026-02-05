"""
Experiment: Effect of dimensionality reduction on complexity metrics.

Tests how PCA dimensionality reduction affects complexity measurements
on high-dimensional real datasets.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import complexity_metrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(dataset_name="Ionosphere", dims=None, plot_datasets=False, terminal_plot=True):
    """
    Run dimensionality reduction experiment.

    Parameters
    ----------
    dataset_name : str
        Dataset to test. Default: "Ionosphere" (34 features)
    dims : list of int
        Target dimensions for PCA. Default: [2, 5, 10, None]
        None means no reduction (original dimensions).
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dimension setting.
    """
    if dims is None:
        dims = [2, 5, 10, None]

    metrics_by_dataset = {}

    for d in dims:
        if d is None:
            label = "original"
            dataset = get_dataset(dataset_name, scale=True)
        else:
            label = f"PCA-{d}"
            dataset = get_dataset(
                dataset_name,
                scale=True,
                dim_reducer="PCA",
                reduce_to_dim=d
            )

        if plot_datasets:
            print(f"\n{label}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = complexity_metrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[label] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Dimensionality Reduction Effect on Complexity")
    print("=" * 50)
    print("Using PCA on Ionosphere dataset (34 features)")
    print()

    metrics = run_experiment(
        dataset_name="Ionosphere",
        dims=[2, 5, 10, None],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
