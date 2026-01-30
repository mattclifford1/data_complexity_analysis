import data_loaders
import math
import matplotlib.pyplot as plt
from data_loaders import get_dataset
from data_loaders.terminal_plots import terminal_show
from metrics import complexity_metrics

    

def plot_metrics(metrics_by_dataset,
                 terminal_plot=False,
                 xlabels_every_subplot=False):
    if not metrics_by_dataset:
        raise ValueError("No datasets found to plot.")

    metric_names = list(next(iter(metrics_by_dataset.values())).keys())
    if not metric_names:
        raise ValueError("No metrics found to plot.")

    dataset_names = list(metrics_by_dataset.keys())
    x_positions = list(range(len(dataset_names)))

    num_metrics = len(metric_names)
    cols = 3
    rows = math.ceil(num_metrics / cols)

    # Auto-size to ~90% of the screen when Tk is available; fall back to a wide layout.
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        fig_w_in = (screen_w * 0.9) / plt.rcParams["figure.dpi"]
        fig_h_in = (screen_h * 0.9) / plt.rcParams["figure.dpi"]
    except ModuleNotFoundError:
        fig_w_in = 4.2 * cols
        fig_h_in = 2.3 * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w_in, fig_h_in), sharex=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, metric_name in enumerate(metric_names):
        ax = axes_list[idx]
        y_values = []
        for dataset_name in dataset_names:
            metrics = metrics_by_dataset[dataset_name]
            if metric_name not in metrics:
                raise KeyError(f"Missing metric {metric_name} for {dataset_name}")
            y_values.append(metrics[metric_name])
        bars = ax.bar(x_positions, y_values)
        for bar, y in zip(bars, y_values):
            ax.annotate(
                f"{y:.2f}",
                (bar.get_x() + bar.get_width() / 2, y),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
            )
        ax.set_title(metric_name)
        ax.grid(True, linestyle="--", alpha=0.35)
        if xlabels_every_subplot or (idx // cols) == (rows - 1):
            ax.set_xticks(x_positions)
            if terminal_plot:
                rotation = 0
            else:
                rotation = 30
            ax.set_xticklabels(dataset_names, rotation=rotation, ha="right")
            # sharex=True hides labels on non-bottom axes by default.
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)

    for ax in axes_list[num_metrics:]:
        ax.axis("off")

    fig.suptitle("Complexity Metrics by Dataset", y=0.99)
    fig.tight_layout(pad=0.8, w_pad=0.6, h_pad=0.8)


    if terminal_plot:
        terminal_show()
    else:
        plt.tight_layout()
        plt.show()

    
if __name__ == "__main__":

    # data_loaders.print_available_datasets()
    all_datasets = data_loaders.get_available_dataset_list()

    datasets_to_test = ['XOR', 'Iris', 'Wine', 'Breast Cancer', 'Moons']
    datasets_to_test = ['Wine', 'Moons', 'Iris']

    # metrics_by_dataset = {}
    # for dataset_name in datasets_to_test:
    #     dataset = get_dataset(dataset_name)
    #     complexity = complexity_metrics(dataset=dataset.get_data_dict())
    #     all_metrics = complexity.get_all_metrics_scalar()
    #     metrics_by_dataset[dataset_name] = all_metrics
    

    # get different Gaussian blobs datasets
    metrics_by_dataset = {}
    for variance in [0.05, 0.1, 0.2, 0.3, 0.5]:
        cov_matrix = [[variance, 0], [0, variance]]
        dataset_name = f'Gaussian_var_{variance}'
        dataset = get_dataset("Gaussian",
                              cov=cov_matrix,
                              name=dataset_name)
        complexity = complexity_metrics(dataset=dataset.get_data_dict())
        all_metrics = complexity.get_all_metrics_scalar()
        metrics_by_dataset[dataset_name] = all_metrics


    plot_metrics(metrics_by_dataset, 
                 xlabels_every_subplot=True,
                 terminal_plot=True)
