"""
Experiment: Effect of class imbalance on complexity metrics.

Varies the ratio of samples between classes.
Imbalance can affect instance-level and structural metrics.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.data_metrics.metrics import ComplexityMetrics
from data_complexity.plotting.plot_multiple_datasets import plot_metrics


def run_experiment(ratios=None, total_samples=400, plot_datasets=False, terminal_plot=True):
    """
    Run class imbalance experiment.

    Parameters
    ----------
    ratios : list of tuple
        (class0, class1) sample ratios to test. Default: [(1,1), (2,1), (4,1), (9,1)]
    total_samples : int
        Approximate total samples (actual may vary due to rounding). Default: 400
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if ratios is None:
        ratios = [(1, 1), (2, 1), (4, 1), (9, 1)]

    metrics_by_dataset = {}

    for ratio in ratios:
        # Calculate sample counts from ratio
        total_ratio = ratio[0] + ratio[1]
        n_class0 = int(total_samples * ratio[0] / total_ratio)
        n_class1 = int(total_samples * ratio[1] / total_ratio)

        dataset_name = f'{ratio[0]}:{ratio[1]}'
        dataset = get_dataset(
            "Gaussian",
            num_samples=[n_class0, n_class1],
            class_separation=4.0,
            cov_type='spherical',
            cov_scale=1.0,
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name} (n={n_class0}+{n_class1}):")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Class Imbalance Effect on Complexity")
    print("=" * 50)
    print("Ratios shown as majority:minority")
    print()

    metrics = run_experiment(
        ratios=[(1, 1), (2, 1), (4, 1), (9, 1)],
        total_samples=400,
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
