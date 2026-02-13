"""
Experiment: Effect of class separation on complexity metrics.

Varies the distance between Gaussian class means while keeping covariance fixed.
Higher separation should result in lower overlap/complexity metrics.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import ComplexityMetrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(separations=None, plot_datasets=False, terminal_plot=True):
    """
    Run class separation experiment.

    Parameters
    ----------
    separations : list of float
        Class separation values to test. Default: [1, 2, 5, 10]
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if separations is None:
        separations = [1, 2, 5, 10]

    metrics_by_dataset = {}

    for sep in separations:
        dataset_name = f'sep={sep}'
        dataset = get_dataset(
            "Gaussian",
            class_separation=sep,
            cov_type='spherical',
            cov_scale=1.0,
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Class Separation Effect on Complexity")
    print("=" * 50)
    print("Hypothesis: Higher separation → lower complexity metrics")
    print()

    metrics = run_experiment(
        separations=[1, 2, 5, 10],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
