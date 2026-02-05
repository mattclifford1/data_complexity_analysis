"""
Experiment: Effect of noise on moons dataset complexity.

The moons dataset creates two interleaving half-circles.
Higher noise blurs the boundary between classes.
"""
import data_loaders
from data_loaders import get_dataset
from data_complexity.metrics import complexity_metrics
from data_complexity.plot_multiple_datasets import plot_metrics


def run_experiment(noise_levels=None, plot_datasets=False, terminal_plot=True):
    """
    Run moons noise experiment.

    Parameters
    ----------
    noise_levels : list of float
        Noise standard deviations to test. Default: [0.05, 0.1, 0.2, 0.4]
    plot_datasets : bool
        Whether to plot each dataset. Default: False
    terminal_plot : bool
        Use terminal plotting. Default: True

    Returns
    -------
    dict
        Metrics by dataset name.
    """
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.2, 0.4]

    metrics_by_dataset = {}

    for noise in noise_levels:
        dataset_name = f'noise={noise}'
        dataset = get_dataset(
            "Moons",
            moons_noise=noise,
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
    print("Experiment: Moons Noise Effect on Complexity")
    print("=" * 50)
    print("Hypothesis: Higher noise → higher complexity metrics")
    print()

    metrics = run_experiment(
        noise_levels=[0.05, 0.1, 0.2, 0.4],
        plot_datasets=True,
        terminal_plot=True
    )

    plot_metrics(
        metrics,
        xlabels_every_subplot=True,
        terminal_plot=True
    )
