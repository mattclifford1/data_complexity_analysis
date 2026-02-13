"""
Experiment: Compare complexity across different synthetic generators.

Compares Gaussian, Moons, Circles, Blobs, and XOR at similar sample sizes
to understand inherent complexity differences between dataset types.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import ComplexityMetrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(num_samples=400, plot_datasets=False, terminal_plot=True):
    """
    Run synthetic generator comparison experiment.

    Parameters
    ----------
    num_samples : int
        Number of samples per dataset. Default: 400
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    datasets_config = [
        ("Gaussian", {"class_separation": 4.0, "cov_type": "spherical"}),
        ("Moons", {"moons_noise": 0.15}),
        ("Circles", {"circles_noise": 0.1}),
        ("Blobs", {"blobs_features": 2}),
        ("XOR", {}),
    ]

    metrics_by_dataset = {}

    for dataset_type, kwargs in datasets_config:
        dataset_name = dataset_type

        # Handle XOR's different num_samples format
        if dataset_type == "XOR":
            kwargs["num_samples"] = [num_samples // 2, num_samples // 2]
        else:
            kwargs["num_samples"] = num_samples

        kwargs["name"] = dataset_name
        dataset = get_dataset(dataset_type, **kwargs)

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Synthetic Generator Complexity Comparison")
    print("=" * 50)
    print("Comparing inherent complexity of different dataset types")
    print()

    metrics = run_experiment(
        num_samples=400,
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
