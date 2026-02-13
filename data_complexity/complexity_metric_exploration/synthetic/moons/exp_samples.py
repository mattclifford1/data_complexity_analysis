"""
Experiment: Effect of sample size on moons dataset complexity.

Tests how the number of samples affects complexity metric estimates.
Sparse sampling may miss boundary details.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import ComplexityMetrics
from data_complexity.plotting.plot_multiple_datasets import plot_metrics


def run_experiment(sample_sizes=None, plot_datasets=False, terminal_plot=True):
    """
    Run moons sample size experiment.

    Parameters
    ----------
    sample_sizes : list of int
        Number of samples to test. Default: [50, 100, 200, 500]
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 500]

    metrics_by_dataset = {}

    for n in sample_sizes:
        dataset_name = f'n={n}'
        dataset = get_dataset(
            "Moons",
            moons_noise=0.15,
            num_samples=n,
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Moons Sample Size Effect on Complexity")
    print("=" * 50)
    print("Fixed noise=0.15, varying sample count")
    print()

    metrics = run_experiment(
        sample_sizes=[50, 100, 200, 500],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
