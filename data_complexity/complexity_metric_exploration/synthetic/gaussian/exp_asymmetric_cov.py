"""
Experiment: Effect of asymmetric class covariances on complexity metrics.

One class has fixed variance while the other varies.
Tests how heterogeneous class distributions affect complexity.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import ComplexityMetrics
from data_complexity.plotting.plot_multiple_datasets import plot_metrics


def run_experiment(class1_scales=None, plot_datasets=False, terminal_plot=True):
    """
    Run asymmetric covariance experiment.

    Class 0 has fixed scale=1.0, class 1 scale varies.

    Parameters
    ----------
    class1_scales : list of float
        Scale values for class 1. Default: [0.5, 1.0, 2.0, 4.0]
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if class1_scales is None:
        class1_scales = [0.5, 1.0, 2.0, 4.0]

    metrics_by_dataset = {}

    for scale1 in class1_scales:
        dataset_name = f'1.0/{scale1}'
        dataset = get_dataset(
            "Gaussian",
            class_separation=4.0,
            cov_type='spherical',
            cov_scale=[1.0, scale1],  # Different scales per class
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Asymmetric Covariance Effect on Complexity")
    print("=" * 50)
    print("Class 0 scale=1.0, Class 1 scale varies")
    print()

    metrics = run_experiment(
        class1_scales=[0.5, 1.0, 2.0, 4.0],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
