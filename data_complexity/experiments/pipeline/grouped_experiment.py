"""
Grouped experiment: run the same sweep on multiple dataset groups, then average
the resulting pairwise distance matrices across groups.

Typical use case: compute "how correlated are complexity metrics across a
parameter sweep?" for several real datasets, then average the answer across
datasets to get a robust, dataset-agnostic result.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from data_complexity.experiments.pipeline.experiment import Experiment
from data_complexity.experiments.pipeline.utils import DatasetSpec, ExperimentConfig
from data_complexity.experiments.plotting import plot_pairwise_heatmap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default aggregation
# ---------------------------------------------------------------------------

def mean_matrices(matrices: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Element-wise mean across a list of N×N DataFrames with aligned indices.

    Only metrics present in *all* matrices are included; a warning is issued
    when some metrics are missing in some groups.

    Parameters
    ----------
    matrices : list of pd.DataFrame
        Each DataFrame is an N×N pairwise distance matrix with the same
        metric names as both index and columns.

    Returns
    -------
    pd.DataFrame
        Element-wise mean over the shared metric set.
    """
    if not matrices:
        raise ValueError("Cannot average an empty list of matrices.")

    all_indices = [set(m.index.tolist()) for m in matrices]
    shared = set.intersection(*all_indices)

    if len(shared) < len(all_indices[0]):
        missing = set.union(*all_indices) - shared
        warnings.warn(
            f"mean_matrices: metrics {missing} are missing in some groups and will be excluded.",
            stacklevel=2,
        )

    shared_sorted = sorted(shared)
    aligned = [m.loc[shared_sorted, shared_sorted] for m in matrices]
    return pd.concat(aligned).groupby(level=0).mean()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GroupedExperimentConfig:
    """
    Configuration for a grouped experiment.

    Parameters
    ----------
    dataset_groups : dict
        Mapping of group_name -> list of DatasetSpec (the sweep for that group).
    base_config : ExperimentConfig
        Template config. Its ``datasets``, ``name``, and ``save_dir`` fields
        are overridden per group.
    aggregation_fn : callable
        Function ``(List[pd.DataFrame]) -> pd.DataFrame`` used to aggregate
        per-group pairwise distance matrices. Default: element-wise mean.
    name : str
        Name for this grouped experiment. Default: "grouped_experiment".
    save_dir : Path, optional
        Root directory to save results. Defaults to ``results/{name}/``
        relative to the current working directory.
    """

    dataset_groups: Dict[str, List[DatasetSpec]]
    base_config: ExperimentConfig
    aggregation_fn: Callable[[List[pd.DataFrame]], pd.DataFrame] = field(
        default=mean_matrices
    )
    name: str = "grouped_experiment"
    save_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.save_dir is None:
            self.save_dir = Path.cwd() / "results" / self.name


# ---------------------------------------------------------------------------
# GroupedExperiment
# ---------------------------------------------------------------------------

class GroupedExperiment:
    """
    Run the same experiment sweep on multiple dataset groups, then aggregate
    the resulting pairwise distance matrices.

    Parameters
    ----------
    config : GroupedExperimentConfig
        Grouped experiment configuration.

    Attributes
    ----------
    experiments : dict
        group_name -> Experiment (populated after :meth:`run`).
    per_group_pairwise_distances : dict
        group_name -> measure_name -> N×N DataFrame
        (populated after :meth:`compute_averaged_pairwise_distances`).
    averaged_pairwise_distances : dict
        measure_name -> N×N DataFrame
        (populated after :meth:`compute_averaged_pairwise_distances`).
    """

    def __init__(self, config: GroupedExperimentConfig) -> None:
        self.config = config
        self.experiments: Dict[str, Experiment] = {}
        self.per_group_pairwise_distances: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.averaged_pairwise_distances: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True, n_jobs: int = 1) -> None:
        """
        Run one Experiment per dataset group.

        Parameters
        ----------
        verbose : bool
            Print progress. Default: True
        n_jobs : int
            Worker count forwarded to each ``Experiment.run()``.
        """
        for group_name, datasets in self.config.dataset_groups.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running group: {group_name}")
                print(f"{'='*60}")

            group_save_dir = self.config.save_dir / "groups" / group_name
            group_config = _clone_config(
                self.config.base_config,
                datasets=datasets,
                name=f"{self.config.name}__{group_name}",
                save_dir=group_save_dir,
            )
            exp = Experiment(group_config)
            exp.run(verbose=verbose, n_jobs=n_jobs)
            self.experiments[group_name] = exp

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def compute_averaged_pairwise_distances(
        self, source: str = "train"
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute pairwise complexity distances per group, then aggregate.

        Calls ``Experiment.compute_complexity_pairwise_distances()`` for each
        group experiment, collects the per-group N×N matrices, and applies
        ``config.aggregation_fn`` per measure.

        Parameters
        ----------
        source : str
            Which complexity data to use: 'train' or 'test'. Default: 'train'

        Returns
        -------
        dict
            measure_name -> aggregated N×N DataFrame.
        """
        if not self.experiments:
            raise RuntimeError("Must call run() before compute_averaged_pairwise_distances().")

        self.per_group_pairwise_distances = {}

        for group_name, exp in self.experiments.items():
            group_matrices = exp.compute_complexity_pairwise_distances(source=source)
            self.per_group_pairwise_distances[group_name] = group_matrices

        # Collect all measure names across groups
        all_measure_names: set[str] = set()
        for group_matrices in self.per_group_pairwise_distances.values():
            all_measure_names.update(group_matrices.keys())

        self.averaged_pairwise_distances = {}
        for measure_name in all_measure_names:
            matrices = [
                self.per_group_pairwise_distances[g][measure_name]
                for g in self.per_group_pairwise_distances
                if measure_name in self.per_group_pairwise_distances[g]
            ]
            if not matrices:
                continue
            if len(matrices) < len(self.experiments):
                missing_groups = [
                    g for g in self.per_group_pairwise_distances
                    if measure_name not in self.per_group_pairwise_distances[g]
                ]
                warnings.warn(
                    f"Measure '{measure_name}' missing in groups {missing_groups}; "
                    "averaging over available groups only.",
                    stacklevel=2,
                )
            self.averaged_pairwise_distances[measure_name] = (
                self.config.aggregation_fn(matrices)
            )

        return self.averaged_pairwise_distances

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self) -> Dict[str, "plt.Figure"]:
        """
        Generate heatmaps of the averaged pairwise distance matrices.

        Returns
        -------
        dict
            measure_name -> matplotlib Figure.
        """
        if not self.averaged_pairwise_distances:
            raise RuntimeError(
                "Must call compute_averaged_pairwise_distances() before plot()."
            )

        measure_display = {
            m.name: m.display_name
            for m in self.config.base_config.pairwise_distance_measures
        }

        figures: Dict[str, plt.Figure] = {}
        for measure_name, matrix in self.averaged_pairwise_distances.items():
            display = measure_display.get(measure_name, measure_name)
            fig = plot_pairwise_heatmap(
                matrix,
                title=f"{self.config.name}: Averaged Complexity Pairwise ({display})",
            )
            figures[measure_name] = fig

        return figures

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, save_dir: Optional[Path] = None) -> None:
        """
        Save averaged matrices as CSVs and heatmap PNGs, then delegate
        per-group saves to each child Experiment.

        Directory layout::

            save_dir/
            ├── data/
            │   └── averaged_pairwise_distances_{measure}.csv
            ├── plots/
            │   └── averaged_pairwise_distances_{measure}.png
            └── groups/
                └── {group_name}/   ← each Experiment saves here

        Parameters
        ----------
        save_dir : Path, optional
            Root directory. Default: config.save_dir
        """
        save_dir = save_dir or self.config.save_dir
        data_dir = save_dir / "data"
        plots_dir = save_dir / "plots"
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save averaged matrices
        for measure_name, matrix in self.averaged_pairwise_distances.items():
            csv_path = data_dir / f"averaged_pairwise_distances_{measure_name}.csv"
            matrix.to_csv(csv_path)

        # Save heatmaps
        figures = self.plot()
        for measure_name, fig in figures.items():
            png_path = plots_dir / f"averaged_pairwise_distances_{measure_name}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Save each group's experiment results
        for group_name, exp in self.experiments.items():
            group_save_dir = save_dir / "groups" / group_name
            exp.save(save_dir=group_save_dir)

        print(f"Saved grouped results to: {save_dir}")
        print(f"  - Averaged CSVs: data/")
        print(f"  - Heatmaps: plots/")
        print(f"  - Per-group: groups/{{group_name}}/")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _clone_config(
    base: ExperimentConfig,
    datasets: List[DatasetSpec],
    name: str,
    save_dir: Path,
) -> ExperimentConfig:
    """Clone base_config, overriding datasets, name, and save_dir."""
    import dataclasses
    return dataclasses.replace(base, datasets=datasets, name=name, save_dir=save_dir)
