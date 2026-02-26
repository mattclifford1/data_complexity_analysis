"""
I/O logic for the Experiment class: save, load, and path resolution.

This module contains save() and load_results() functions for persisting
experiment results to disk and loading them back. It is designed to be
called from Experiment.save() and Experiment.load_results() via thin delegation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import pandas as pd

from data_complexity.experiments.classification import get_default_models
from data_complexity.experiments.pipeline.utils import (
    ExperimentResultsContainer,
    PlotType,
    make_json_safe_dict,
)

if TYPE_CHECKING:
    from data_complexity.experiments.pipeline.experiment import Experiment


def _resolve_path(save_dir: Path, filename: str, subfolder: Optional[str] = None) -> Path:
    """
    Resolve file path, checking new structure first, then falling back to legacy flat structure.

    Parameters
    ----------
    save_dir : Path
        Base directory to search in.
    filename : str
        Name of the file to find.
    subfolder : str, optional
        Subfolder name (e.g., "data", "plots", "datasets").

    Returns
    -------
    Path
        Resolved path to the file.
    """
    if subfolder:
        new_path = save_dir / subfolder / filename
        if new_path.exists():
            return new_path

    # Fall back to flat structure for backwards compatibility
    old_path = save_dir / filename
    if old_path.exists():
        return old_path

    # Return new path (will raise FileNotFoundError if neither exists)
    return save_dir / subfolder / filename if subfolder else old_path


def save(experiment: "Experiment", save_dir: Optional[Path] = None) -> None:
    """
    Save results to CSVs, plots to PNGs, and experiment metadata to JSON.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance providing results, config, and datasets.
    save_dir : Path, optional
        Directory to save to. Default: config.save_dir
    """
    if experiment.results is None:
        raise RuntimeError("Must run experiment before saving.")

    save_dir = save_dir or experiment.config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create subfolders for organized results
    data_dir = save_dir / "data"
    plots_dir = save_dir / f"plots-{experiment.config.name}"
    datasets_dir = save_dir / "datasets"

    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    datasets_dir.mkdir(exist_ok=True)

    # Save experiment metadata
    models = experiment.config.models or get_default_models()
    metadata = {
        "experiment_name": experiment.config.name,
        "datasets": [
            {
                "label": spec.label,
                "type": spec.dataset_type,
                "params": make_json_safe_dict(spec.fixed_params),
            }
            for spec in experiment.config.datasets
        ],
        "x_label": experiment.config.x_label,
        "ml_models": [
            {
                "name": model.name,
                "class": model.__class__.__name__,
                "parameters": {
                    k: v for k, v in model.__dict__.items()
                    if not k.startswith("_") and k not in ["model_params"]
                },
            }
            for model in models
        ],
        "ml_metrics": experiment.config.ml_metrics,
        "cv_folds": experiment.config.cv_folds,
        "correlation_target": experiment.config.correlation_target,
        "plots": [pt.name for pt in experiment.config.plots],
        "run_mode": experiment.config.run_mode.value,
        "pairwise_distance_measures": [m.name for m in experiment.config.pairwise_distance_measures],
    }

    with open(save_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save backward-compat CSVs (train complexity + test ML)
    if experiment.results.complexity_df is not None:
        experiment.results.complexity_df.to_csv(data_dir / "complexity_metrics.csv", index=False)
    if experiment.results.ml_df is not None:
        experiment.results.ml_df.to_csv(data_dir / "ml_performance.csv", index=False)

    # Save train/test split CSVs if available
    if experiment.results.train_complexity_df is not None:
        experiment.results.train_complexity_df.to_csv(
            data_dir / "train_complexity_metrics.csv", index=False
        )
    if experiment.results.test_complexity_df is not None:
        experiment.results.test_complexity_df.to_csv(
            data_dir / "test_complexity_metrics.csv", index=False
        )
    if experiment.results.train_ml_df is not None:
        experiment.results.train_ml_df.to_csv(
            data_dir / "train_ml_performance.csv", index=False
        )
    if experiment.results.test_ml_df is not None:
        experiment.results.test_ml_df.to_csv(
            data_dir / "test_ml_performance.csv", index=False
        )

    if experiment.results.distances_df is not None:
        experiment.results.distances_df.to_csv(data_dir / "distances.csv", index=False)

    for slug, matrix in experiment.results.complexity_pairwise_distances.items():
        matrix.to_csv(data_dir / f"complexity_pairwise_distances_{slug}.csv")

    for slug, matrix in experiment.results.complexity_pairwise_distances_test.items():
        matrix.to_csv(data_dir / f"complexity_pairwise_distances_test_{slug}.csv")

    for slug, matrix in experiment.results.ml_pairwise_distances.items():
        matrix.to_csv(data_dir / f"ml_pairwise_distances_{slug}.csv")

    if experiment.results.per_classifier_distances_df is not None:
        experiment.results.per_classifier_distances_df.to_csv(
            data_dir / "per_classifier_distances.csv", index=False
        )

    # Save plots to plots/ subfolder (sub-keyed figures may include path separators)
    figures = experiment.plot()
    for plot_type, fig in figures.items():
        stem = plot_type.name.lower() if isinstance(plot_type, PlotType) else plot_type
        filepath = plots_dir / f"{stem}.png"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Save dataset visualizations to datasets/ subfolder
    # Create dual plot: full dataset + train/test split
    for label, dataset in experiment.datasets.items():
        # Try to create train/test split plot
        # Some datasets may not support splitting (e.g., minority_reduce_scaler=1)
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Left: Full dataset
            dataset.plot_dataset(ax=axes[0])
            axes[0].set_title("Full Dataset", fontsize=12, fontweight='bold')

            # Middle & Right: Train/test split
            dataset.plot_train_test_split(ax=(axes[1], axes[2]))

            # Add overall title with dataset label
            fig.suptitle(f"Dataset: {label}", fontsize=14, fontweight='bold', y=1.02)

        except (ValueError, AttributeError):
            # Fallback: single plot if train/test split not supported
            plt.close(fig)  # Close the 3-panel figure
            fig, ax = plt.subplots(figsize=(8, 6))
            dataset.plot_dataset(ax=ax)
            ax.set_title(f"Full Dataset: {label}", fontsize=12, fontweight='bold')

        # Sanitize label for filename (replace = with _)
        safe_label = label.replace("=", "_").replace(" ", "_")
        filename = f"dataset_{safe_label}.png"
        fig.savefig(datasets_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved results to: {save_dir}")
    print(f"  - Metadata: experiment_metadata.json")
    print(f"  - Data CSVs: data/")
    print(f"  - Plots: plots-{experiment.config.name}/")
    print(f"  - Datasets: datasets/")


def load_results(
    experiment: "Experiment",
    save_dir: Optional[Path] = None,
) -> ExperimentResultsContainer:
    """
    Load previously saved results from CSVs.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance whose results attribute will be populated.
    save_dir : Path, optional
        Directory to load from. Default: config.save_dir

    Returns
    -------
    ExperimentResultsContainer
        Loaded results.
    """
    save_dir = save_dir or experiment.config.save_dir

    # Resolve paths with backwards compatibility for flat structure
    complexity_path = _resolve_path(save_dir, "complexity_metrics.csv", "data")
    ml_path = _resolve_path(save_dir, "ml_performance.csv", "data")
    corr_path = _resolve_path(save_dir, "distances.csv", "data")

    experiment.results = ExperimentResultsContainer(experiment.config)
    experiment.results._complexity_df = pd.read_csv(complexity_path)
    experiment.results._ml_df = pd.read_csv(ml_path)

    if corr_path.exists():
        experiment.results._distances_df = pd.read_csv(corr_path)

    # Load train/test CSVs if present
    train_complexity_path = _resolve_path(
        save_dir, "train_complexity_metrics.csv", "data"
    )
    if train_complexity_path.exists():
        experiment.results._train_complexity_df = pd.read_csv(train_complexity_path)

    test_complexity_path = _resolve_path(
        save_dir, "test_complexity_metrics.csv", "data"
    )
    if test_complexity_path.exists():
        experiment.results._test_complexity_df = pd.read_csv(test_complexity_path)

    train_ml_path = _resolve_path(
        save_dir, "train_ml_performance.csv", "data"
    )
    if train_ml_path.exists():
        experiment.results._train_ml_df = pd.read_csv(train_ml_path)

    test_ml_path = _resolve_path(
        save_dir, "test_ml_performance.csv", "data"
    )
    if test_ml_path.exists():
        experiment.results._test_ml_df = pd.read_csv(test_ml_path)

    per_classifier_dist_path = _resolve_path(
        save_dir, "per_classifier_distances.csv", "data"
    )
    if per_classifier_dist_path.exists():
        experiment.results._per_classifier_distances_df = pd.read_csv(per_classifier_dist_path)

    experiment.datasets = {}  # Clear since we don't have loaders for loaded results

    return experiment.results
