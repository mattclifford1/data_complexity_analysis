"""
Experiment: Compare complexity across real-world datasets.

Compares complexity metrics for various UCI and other real datasets
to understand how real data differs from synthetic.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.data_metrics.metrics import ComplexityMetrics
from data_complexity.plotting.plot_multiple_datasets import plot_metrics


def run_experiment(dataset_names=None, plot_datasets=False, terminal_plot=True):
    """
    Run real dataset comparison experiment.

    Parameters
    ----------
    dataset_names : list of str
        Dataset names to compare. Default: common UCI datasets
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if dataset_names is None:
        dataset_names = [
            "Iris",
            "Wine",
            "Breast Cancer",
            "Banknote",
            "Ionosphere",
        ]

    metrics_by_dataset = {}

    for name in dataset_names:
        try:
            dataset = get_dataset(name)

            if plot_datasets:
                print(f"\n{name}:")
                dataset.plot_dataset(terminal_plot=terminal_plot)

            complexity = ComplexityMetrics(dataset=dataset.get_data_dict())
            metrics_by_dataset[name] = complexity.get_all_metrics_scalar()

        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")
            continue

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Real Dataset Complexity Comparison")
    print("=" * 50)
    print("Comparing complexity metrics across UCI datasets")
    print()

    metrics = run_experiment(
        dataset_names=["Iris", "Wine", "Breast Cancer", "Banknote", "Ionosphere"],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
