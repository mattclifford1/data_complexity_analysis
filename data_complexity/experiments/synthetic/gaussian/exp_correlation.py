"""
Experiment: Effect of feature correlation on complexity metrics.

Varies the correlation strength in symmetric covariance matrices.
Correlation changes the shape of class distributions from circular to elliptical.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import complexity_metrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(correlations=None, plot_datasets=False, terminal_plot=True):
    """
    Run feature correlation experiment.

    Parameters
    ----------
    correlations : list of float
        Correlation values to test (-1 to 1). Default: [-0.8, -0.3, 0.0, 0.3, 0.8]
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if correlations is None:
        correlations = [-0.8, -0.3, 0.0, 0.3, 0.8]

    metrics_by_dataset = {}

    for corr in correlations:
        dataset_name = f'corr={corr}'
        dataset = get_dataset(
            "Gaussian",
            class_separation=4.0,
            cov_type='symmetric',
            cov_scale=1.0,
            cov_correlation=corr,
            name=dataset_name
        )

        if plot_datasets:
            print(f"\n{dataset_name}:")
            dataset.plot_dataset(terminal_plot=terminal_plot)

        complexity = complexity_metrics(dataset=dataset.get_data_dict())
        metrics_by_dataset[dataset_name] = complexity.get_all_metrics_scalar()

    return metrics_by_dataset


if __name__ == "__main__":
    print("Experiment: Feature Correlation Effect on Complexity")
    print("=" * 50)
    print("Correlation changes distribution shape from circular to elliptical")
    print()

    metrics = run_experiment(
        correlations=[-0.8, -0.3, 0.0, 0.3, 0.8],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
