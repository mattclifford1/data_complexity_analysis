"""
Experiment: Effect of covariance scale (variance) on complexity metrics.

Varies the scale of covariance matrices while keeping class separation fixed.
Higher variance causes more class overlap, increasing complexity.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import ComplexityMetrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(scales=None, plot_datasets=False, terminal_plot=True):
    """
    Run covariance scale experiment.

    Parameters
    ----------
    scales : list of float
        Covariance scale values to test. Default: [0.5, 1.0, 2.0, 4.0]
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if scales is None:
        scales = [0.5, 1.0, 2.0, 4.0]

    metrics_by_dataset = {}

    for scale in scales:
        dataset_name = f'scale={scale}'
        dataset = get_dataset(
            "Gaussian",
            class_separation=4.0,
            cov_type='spherical',
            cov_scale=scale,
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Covariance Scale Effect on Complexity")
    print("=" * 50)
    print("Hypothesis: Higher scale (variance) → higher complexity metrics")
    print()

    metrics = run_experiment(
        scales=[0.5, 1.0, 2.0, 4.0],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
