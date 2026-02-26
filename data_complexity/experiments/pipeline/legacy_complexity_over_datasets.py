"""
ComplexityCollection: Multi-dataset complexity analysis.

Collects complexity metrics across a mix of real and synthetic datasets,
averages over multiple random seeds, and computes pairwise correlations
between complexity metrics across the dataset collection.
"""
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_complexity.data_metrics.metrics import ComplexityMetrics
from data_complexity.experiments.pipeline.utils import (
    _average_dicts,
)
from data_complexity.experiments.plotting import plot_complexity_correlations_heatmap


@dataclass
class DatasetEntry:
    """
    A single entry in the complexity collection.

    Parameters
    ----------
    name : str
        Human-readable name for this dataset.
    data : dict, optional
        Pre-loaded data dict with 'X' and 'y' keys. Set for real datasets.
    dataset_type : str, optional
        Synthetic dataset type name (e.g., 'Gaussian', 'Moons').
        Set for synthetic datasets.
    dataset_params : dict
        Parameters passed to get_dataset() for synthetic datasets.
    """

    name: str
    data: Optional[Dict[str, Any]] = None
    dataset_type: Optional[str] = None
    dataset_params: Dict[str, Any] = field(default_factory=dict)


class ComplexityCollection:
    """
    Multi-dataset complexity analysis collection.

    Collects complexity metrics across a mix of real and synthetic datasets,
    averages over multiple random seeds, and computes pairwise correlations
    between complexity metrics. The result is a (n_datasets × n_metrics)
    DataFrame, which can then be correlated column-wise to understand how
    complexity measures relate to each other across a dataset benchmark.

    ``save()`` writes CSVs, a correlation heatmap, per-dataset PNG plots in a
    ``datasets/`` subdirectory, and a ``datasets_overview.png`` grid. Synthetic
    datasets produce a 3-panel figure (Full | Train | Test); real (pre-loaded)
    datasets produce a 2-feature scatter plot.

    Parameters
    ----------
    seeds : int
        Number of random seeds for train/test splitting. Default: 5
    train_size : float
        Fraction of data to use for training in each split. Default: 0.5

    Examples
    --------
    >>> from data_complexity.experiments import ComplexityCollection
    >>> collection = (
    ...     ComplexityCollection(seeds=5, train_size=0.5)
    ...     .add_dataset("iris", {"X": X_iris, "y": y_iris})
    ...     .add_synthetic("easy_gaussian", "Gaussian", {"class_separation": 4.0})
    ...     .add_synthetic_sweep(
    ...         base_name="moons",
    ...         dataset_type="Moons",
    ...         fixed_params={"n_samples": 500},
    ...         vary_param="noise",
    ...         values=[0.05, 0.1, 0.2, 0.3],
    ...     )
    ... )
    >>> metrics_df = collection.compute()          # (n_datasets × n_metrics)
    >>> corr_matrix = collection.compute_correlations()  # (n_metrics × n_metrics)
    >>> fig = collection.plot_heatmap()
    >>> collection.save(Path("results/my_study/"))
    # Saves: complexity_metrics.csv, complexity_correlations.csv,
    #        complexity_correlations_heatmap.png, datasets_overview.png,
    #        datasets/dataset_<name>.png  (one per entry)
    """

    def __init__(self, 
                 seeds: int = 5, 
                 train_size: float = 0.5,
                 name: str = 'complexity_collection',
                 ) -> None:
        self.seeds = seeds
        self.train_size = train_size
        self.name = name
        self._entries: List[DatasetEntry] = []
        self._metrics_df: Optional[pd.DataFrame] = None
        self._correlations_df: Optional[pd.DataFrame] = None
        self._datasets: Dict[str, Any] = {}

    def add_dataset(self, name: str, data: Dict[str, Any]) -> "ComplexityCollection":
        """
        Add a pre-loaded dataset.

        Parameters
        ----------
        name : str
            Human-readable name for this entry.
        data : dict
            Data dict with 'X' (feature array) and 'y' (label array) keys.

        Returns
        -------
        ComplexityCollection
            Self, for fluent chaining.
        """
        self._entries.append(DatasetEntry(name=name, data=data))
        self._invalidate_cache()
        return self

    def add_synthetic(
        self,
        name: str,
        dataset_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "ComplexityCollection":
        """
        Add a single synthetic dataset with fixed parameters.

        Parameters
        ----------
        name : str
            Human-readable name for this entry.
        dataset_type : str
            Dataset type passed to ``get_dataset()``, e.g. 'Gaussian', 'Moons'.
        params : dict, optional
            Additional keyword arguments forwarded to ``get_dataset()``. Default: {}

        Returns
        -------
        ComplexityCollection
            Self, for fluent chaining.
        """
        self._entries.append(
            DatasetEntry(
                name=name,
                dataset_type=dataset_type,
                dataset_params=params or {},
            )
        )
        self._invalidate_cache()
        return self

    def add_synthetic_sweep(
        self,
        base_name: str,
        dataset_type: str,
        fixed_params: Optional[Dict[str, Any]],
        vary_param: str,
        values: List[Any],
        name_format: str = "{base}_{param}={value}",
    ) -> "ComplexityCollection":
        """
        Add one synthetic dataset entry per parameter value.

        Parameters
        ----------
        base_name : str
            Base name prefix used in ``name_format``.
        dataset_type : str
            Dataset type passed to ``get_dataset()``, e.g. 'Gaussian', 'Moons'.
        fixed_params : dict, optional
            Parameters that remain constant across all sweep entries.
        vary_param : str
            Name of the parameter to sweep over.
        values : list
            Values for the varied parameter.
        name_format : str
            Format string for entry names. Supports ``{base}``, ``{param}``,
            and ``{value}`` placeholders. Default: ``"{base}_{param}={value}"``

        Returns
        -------
        ComplexityCollection
            Self, for fluent chaining.
        """
        fixed = fixed_params or {}
        for value in values:
            entry_name = name_format.format(base=base_name, param=vary_param, value=value)
            params = {**fixed, vary_param: value}
            self._entries.append(
                DatasetEntry(
                    name=entry_name,
                    dataset_type=dataset_type,
                    dataset_params=params,
                )
            )
        self._invalidate_cache()
        return self

    def _invalidate_cache(self) -> None:
        """Clear cached computed results when entries change."""
        self._metrics_df = None
        self._correlations_df = None
        self._datasets = {}

    def _get_representative_dataset(self, entry: DatasetEntry) -> Any:
        """
        Return one dataset object (or raw dict) for plotting.

        For pre-loaded real datasets, returns the raw data dict.
        For synthetic datasets, calls ``get_dataset()`` and returns the dataset
        object (which has ``plot_dataset`` / ``plot_train_test_split``).

        Parameters
        ----------
        entry : DatasetEntry
            The dataset entry.

        Returns
        -------
        Any
            Dataset object (synthetic) or raw dict (real).
        """
        if entry.data is not None:
            return entry.data

        from data_loaders import get_dataset  # type: ignore

        return get_dataset(
            dataset_name=entry.dataset_type,
            train_size=self.train_size,
            **entry.dataset_params,
        )

    def _plot_single_dataset(self, name: str, dataset_or_data: Any) -> plt.Figure:
        """
        Create a visualization figure for one dataset entry.

        Synthetic datasets (those with a ``plot_dataset`` attribute) get a
        3-panel layout: Full | Train | Test.  Real datasets (raw dicts) get a
        single scatter panel of the first two features.

        Parameters
        ----------
        name : str
            Dataset name used as the figure title.
        dataset_or_data : Any
            Either a dataset object (synthetic) or a raw dict with 'X'/'y'.

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        """
        if hasattr(dataset_or_data, "plot_dataset"):
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                dataset_or_data.plot_dataset(ax=axes[0])
                axes[0].set_title("Full Dataset", fontsize=12, fontweight="bold")
                dataset_or_data.plot_train_test_split(ax=(axes[1], axes[2]))
                fig.suptitle(f"Dataset: {name}", fontsize=14, fontweight="bold", y=1.02)
            except (ValueError, AttributeError):
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8, 6))
                dataset_or_data.plot_dataset(ax=ax)
                ax.set_title(f"Full Dataset: {name}", fontsize=12, fontweight="bold")
        else:
            X = dataset_or_data["X"]
            y = dataset_or_data["y"]
            fig, ax = plt.subplots(figsize=(8, 6))
            for label in np.unique(y):
                mask = y == label
                ax.scatter(X[mask, 0], X[mask, 1], label=f"Class {label}", alpha=0.6, s=20)
            ax.set_title(f"Dataset: {name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Feature 0")
            ax.set_ylabel("Feature 1")
            ax.legend()

        return fig

    def _plot_datasets_overview(self) -> plt.Figure:
        """
        Create a grid overview of all datasets (one subplot each).

        Synthetic datasets use ``plot_dataset()``. Real datasets get a scatter
        of the first two features.

        Returns
        -------
        plt.Figure
            The composite overview figure.
        """
        n_datasets = len(self._datasets)
        n_cols = min(4, n_datasets)
        n_rows = math.ceil(n_datasets / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        flat_axes = np.array(axes).flatten() if n_datasets > 1 else [axes]

        for idx, (name, dataset_or_data) in enumerate(self._datasets.items()):
            ax = flat_axes[idx]
            if hasattr(dataset_or_data, "plot_dataset"):
                try:
                    dataset_or_data.plot_dataset(ax=ax)
                except Exception:
                    ax.text(0.5, 0.5, name, ha="center", va="center", transform=ax.transAxes)
            else:
                X = dataset_or_data["X"]
                y = dataset_or_data["y"]
                for label in np.unique(y):
                    mask = y == label
                    ax.scatter(X[mask, 0], X[mask, 1], label=f"Class {label}", alpha=0.6, s=15)
            ax.set_title(name, fontsize=10, fontweight="bold")

        for idx in range(n_datasets, len(flat_axes)):
            flat_axes[idx].axis("off")

        fig.suptitle("All Datasets", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def _save_dataset_plots(self, path: Path) -> None:
        """
        Save per-dataset plots and a combined overview to ``path``.

        Creates a ``datasets/`` subdirectory under ``path`` containing one PNG
        per entry.  Also writes ``datasets_overview.png`` directly to ``path``.

        Parameters
        ----------
        path : Path
            Output directory (must already exist).
        """
        datasets_dir = path / "datasets"
        plots_dir = path / "plots"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        for name, dataset_or_data in self._datasets.items():
            safe_name = name.replace("=", "_").replace(" ", "_")
            fig = self._plot_single_dataset(name, dataset_or_data)
            fig.savefig(datasets_dir / f"dataset_{safe_name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig = self._plot_datasets_overview()

        fig.savefig(plots_dir / "datasets_overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _compute_entry(self, entry: DatasetEntry) -> Dict[str, float]:
        """
        Compute average complexity metrics for one entry across all seeds.

        For pre-loaded datasets, uses ``proportional_split`` with a deep copy
        of the data dict on each seed (as ``proportional_split`` may mutate
        the input). For synthetic datasets, uses ``get_dataset`` followed by
        ``get_train_test_split``.

        Parameters
        ----------
        entry : DatasetEntry
            The dataset entry to compute metrics for.

        Returns
        -------
        dict
            Complexity metric name → mean value across seeds.
        """
        from data_loaders import get_dataset, proportional_split  # type: ignore

        seed_results: List[Dict[str, float]] = []
        for seed_i in range(self.seeds):
            seed = 42 + seed_i
            if entry.data is not None:
                # Pre-loaded dataset: deep copy to avoid mutating original
                data_copy = copy.deepcopy(entry.data)
                train_data, _ = proportional_split(
                    data_copy, train_size=self.train_size, seed=seed
                )
            else:
                # Synthetic dataset: generate and split
                dataset = get_dataset(
                    dataset_name=entry.dataset_type,
                    train_size=self.train_size,
                    **entry.dataset_params,
                )
                train_data, _ = dataset.get_train_test_split(seed=seed)

            metrics = ComplexityMetrics(dataset=train_data).get_all_metrics_scalar()
            seed_results.append(metrics)

        return _average_dicts(seed_results)

    def compute(self) -> pd.DataFrame:
        """
        Compute complexity metrics for all entries.

        Runs ``seeds`` random train/test splits for each entry, computes
        complexity on the train split, and averages across seeds. The result
        is cached; call again or add more entries to recompute.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per dataset entry and one column per
            complexity metric. The index is the dataset name.
        """
        rows = []
        self._datasets = {}
        for entry in self._entries:
            avg_metrics = self._compute_entry(entry)
            row = {"dataset": entry.name, **avg_metrics}
            rows.append(row)
            self._datasets[entry.name] = self._get_representative_dataset(entry)

        self._metrics_df = pd.DataFrame(rows).set_index("dataset")
        self._correlations_df = None  # Invalidate correlation cache
        return self._metrics_df

    def compute_correlations(self) -> pd.DataFrame:
        """
        Compute pairwise Pearson correlations between complexity metrics.

        Constant columns (zero standard deviation across datasets) are dropped
        automatically, as they produce NaN correlations. Calls ``compute()``
        first if metrics have not been computed yet.

        Returns
        -------
        pd.DataFrame
            N×N symmetric Pearson correlation matrix, indexed and columned
            by metric name.
        """
        if self._metrics_df is None:
            self.compute()

        # Drop constant columns (std == 0 produces NaN correlations)
        valid_cols = [c for c in self._metrics_df.columns if self._metrics_df[c].std() > 0]
        self._correlations_df = self._metrics_df[valid_cols].corr(method="pearson")
        return self._correlations_df

    def plot_heatmap(
        self,
        title: str = "Complexity Metric Correlations",
    ) -> plt.Figure:
        """
        Plot a heatmap of pairwise Pearson correlations between complexity metrics.

        Calls ``compute_correlations()`` first if not already computed.

        Parameters
        ----------
        title : str
            Plot title. Default: "Complexity Metric Correlations"

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        """
        if self._correlations_df is None:
            self.compute_correlations()

        return plot_complexity_correlations_heatmap(
            corr_matrix=self._correlations_df,
            title=title,
        )

    def save(self, path: Path) -> None:
        """
        Save results to CSV files, a heatmap PNG, and dataset visualizations.

        Writes the following files to ``path``:

        - ``complexity_metrics.csv`` — per-dataset complexity metrics
          (n_datasets × n_metrics, indexed by dataset name)
        - ``complexity_correlations.csv`` — N×N Pearson correlation matrix
        - ``complexity_correlations_heatmap.png`` — heatmap visualization
        - ``datasets/`` — subdirectory with one PNG per dataset entry
        - ``datasets_overview.png`` — grid overview of all datasets

        Calls ``compute()`` and ``compute_correlations()`` automatically if
        results have not been computed yet.

        Parameters
        ----------
        path : Path
            Directory to save results to. Created if it does not exist.
        """
        path = Path(path)
        data_dir = path / "data"
        plots_dir = path / "plots"
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        if self._metrics_df is None:
            self.compute()
        if self._correlations_df is None:
            self.compute_correlations()

        self._metrics_df.to_csv(data_dir / "complexity_metrics.csv")
        self._correlations_df.to_csv(data_dir / "complexity_correlations.csv")

        fig = self.plot_heatmap(title=f"{self.name}")
        fig.savefig(
            plots_dir / "complexity_correlations_heatmap.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        self._save_dataset_plots(path)

        n_datasets = len(self._metrics_df)
        n_metrics = len(self._metrics_df.columns)
        n_corr = len(self._correlations_df)
        print(f"Saved to: {path}")
        print(f"  - complexity_metrics.csv ({n_datasets} datasets × {n_metrics} metrics)")
        print(f"  - complexity_correlations.csv ({n_corr}×{n_corr} matrix)")
        print(f"  - complexity_correlations_heatmap.png")
        print(f"  - datasets/   ({n_datasets} dataset plots)")
        print(f"  - datasets_overview.png")
