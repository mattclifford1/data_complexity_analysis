"""
Experiment class for complexity vs ML performance analysis.

The actual run, plot, distance, and I/O logic lives in the sibling modules:
- runner.py    — experiment loop and parallel worker
- plotting.py  — visualization generation
- distances.py — distance/association computation
- io.py        — save/load to/from disk
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from data_complexity.experiments.pipeline import (
    runner,
    plotting as _plotting,
    distances as _distances,
    io,
)
from data_complexity.experiments.pipeline.metric_distance import (
    DistanceBetweenMetrics,
    PearsonCorrelation,
)
from data_complexity.experiments.pipeline.utils import (
    ExperimentConfig,
    ExperimentResultsContainer,
    PlotType,
    RunMode,
)

if __name__ != "__main__":
    import matplotlib.pyplot as _plt  # noqa: F401  (TYPE_CHECKING-style guard not needed here)


class Experiment:
    """
    Main experiment runner for complexity vs ML performance analysis.

    Uses train/test splits: for each parameter value and random seed, the data
    is split into train and test sets. Complexity metrics are computed on both,
    and ML models are trained on train and evaluated on both.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Optional[ExperimentResultsContainer] = None
        self._get_dataset = None  # Lazy loader for dataset function to avoid circular imports
        self.datasets: Dict[Any, Any] = {}  # Store loaders for visualization

    def _load_dataset_loader(self):
        """Lazy load data_loaders to avoid import at module level."""
        if self._get_dataset is None:
            import data_loaders  # type: ignore
            from data_loaders import get_dataset  # type: ignore

            self._get_dataset = get_dataset

    def run(self, verbose: bool = True, n_jobs: int = 1) -> ExperimentResultsContainer:
        """
        Execute the experiment loop with train/test splits.

        For each parameter value, generates a dataset, then for each random
        seed (controlled by ``cv_folds``), splits into train/test using the
        dataset's built-in proportional_split(), computes complexity on both,
        and trains ML models on train to evaluate on both. Results are averaged
        across seeds.

        Parameters
        ----------
        verbose : bool
            Print progress. Default: True
        n_jobs : int
            Number of parallel worker processes for the parameter loop.
            Follows sklearn convention:

            - ``1`` (default) — sequential execution, unchanged behaviour.
            - ``N > 1`` — N worker processes via ``ProcessPoolExecutor``.
            - ``-1`` — one process per CPU (executor chooses ``max_workers``).

        Returns
        -------
        ExperimentResultsContainer
            Results container with train/test complexity and ML DataFrames.
        """
        return runner.run(self, verbose=verbose, n_jobs=n_jobs)

    def compute_distances(
        self,
        ml_column: Optional[str] = None,
        complexity_source: str = "train",
        ml_source: str = "test",
        distance: DistanceBetweenMetrics = PearsonCorrelation(),
    ) -> pd.DataFrame:
        """
        Compute distances between complexity metrics and ML performance.

        Parameters
        ----------
        ml_column : str, optional
            ML metric column to measure against. Default: config.correlation_target
        complexity_source : str
            Which complexity to use: 'train' or 'test'. Default: 'train'
        ml_source : str
            Which ML results to use: 'train' or 'test'. Default: 'test'
        distance : DistanceBetweenMetrics
            Distance/association measure to use. Default: PearsonCorrelation()

        Returns
        -------
        pd.DataFrame
            Results sorted by abs_distance descending.
        """
        return _distances.compute_distances(
            self,
            ml_column=ml_column,
            complexity_source=complexity_source,
            ml_source=ml_source,
            distance=distance,
        )

    def compute_complexity_pairwise_distances(
        self,
        source: str = "train",
        distances: Optional[List[DistanceBetweenMetrics]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute pairwise distances between complexity metrics for all requested measures.

        Always computes for both train and test data (when available).
        Results stored in ``results.complexity_pairwise_distances`` and
        ``results.complexity_pairwise_distances_test``, keyed by measure slug.

        Parameters
        ----------
        source : str
            Which dict to return: 'train' or 'test'. Default: 'train'
        distances : list of DistanceBetweenMetrics, optional
            Measures to compute. Default: config.pairwise_distance_measures

        Returns
        -------
        dict
            slug -> N×N symmetric DataFrame for the requested source.
        """
        return _distances.compute_complexity_pairwise_distances(
            self, source=source, distances=distances
        )

    def compute_ml_pairwise_distances(
        self,
        source: str = "test",
        distances: Optional[List[DistanceBetweenMetrics]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute pairwise distances between ML performance metrics for all requested measures.

        Results stored in ``results.ml_pairwise_distances``, keyed by measure slug.

        Parameters
        ----------
        source : str
            Which ML data to use: 'train' or 'test'. Default: 'test'
        distances : list of DistanceBetweenMetrics, optional
            Measures to compute. Default: config.pairwise_distance_measures

        Returns
        -------
        dict
            slug -> N×N symmetric DataFrame.
        """
        return _distances.compute_ml_pairwise_distances(self, source=source, distances=distances)

    def compute_all_distances(
        self,
        distance: DistanceBetweenMetrics = PearsonCorrelation(),
    ) -> pd.DataFrame:
        """
        Compute distances for all ML metric columns.

        Parameters
        ----------
        distance : DistanceBetweenMetrics
            Distance/association measure to use. Default: PearsonCorrelation()

        Returns
        -------
        pd.DataFrame
            All distances combined.
        """
        return _distances.compute_all_distances(self, distance=distance)

    def compute_per_classifier_distances(
        self,
        complexity_source: str = "train",
        ml_source: str = "test",
        distance: DistanceBetweenMetrics = PearsonCorrelation(),
    ) -> pd.DataFrame:
        """
        Compute complexity-vs-ML distances aggregated per metric across classifiers.

        For each ML metric type (e.g., 'accuracy', 'f1'), computes the distance
        between each complexity metric and every individual classifier's column,
        then reports mean and std of those values.

        Parameters
        ----------
        complexity_source : str
            'train' or 'test'. Default: 'train'
        ml_source : str
            'train' or 'test'. Default: 'test'
        distance : DistanceBetweenMetrics
            Distance/association measure to use. Default: PearsonCorrelation()

        Returns
        -------
        pd.DataFrame
            Columns: complexity_metric, ml_metric, mean_distance,
                     std_distance, abs_mean_distance.
            Sorted by abs_mean_distance descending.
        """
        return _distances.compute_per_classifier_distances(
            self,
            complexity_source=complexity_source,
            ml_source=ml_source,
            distance=distance,
        )

    def _compute_single_distance(self, ml_column: str) -> pd.DataFrame:
        """Compute distances for a single ML column."""
        return _distances._compute_single_distance(self, ml_column)

    def plot(
        self, plot_types: Optional[List[PlotType]] = None
    ) -> Dict[Union[PlotType, str], "_plt.Figure"]:
        """
        Generate experiment plots.

        Parameters
        ----------
        plot_types : list of PlotType, optional
            Plot types to generate. Default: config.plots

        Returns
        -------
        dict
            PlotType or str -> matplotlib Figure. String keys are used when a single
            PlotType generates multiple figures (e.g. COMPLEXITY_CORRELATIONS produces
            'complexity_pairwise_distances_train' and 'complexity_pairwise_distances_test').
        """
        return _plotting.plot(self, plot_types=plot_types)

    def save(self, save_dir: Optional[Path] = None) -> None:
        """
        Save results to CSVs, plots to PNGs, and experiment metadata to JSON.

        Parameters
        ----------
        save_dir : Path, optional
            Directory to save to. Default: config.save_dir
        """
        io.save(self, save_dir=save_dir)

    def load_results(self, save_dir: Optional[Path] = None) -> ExperimentResultsContainer:
        """
        Load previously saved results from CSVs.

        Parameters
        ----------
        save_dir : Path, optional
            Directory to load from. Default: config.save_dir

        Returns
        -------
        ExperimentResultsContainer
            Loaded results.
        """
        return io.load_results(self, save_dir=save_dir)

    def print_summary(self, top_n: int = 10) -> None:
        """
        Print top correlations summary.

        Parameters
        ----------
        top_n : int
            Number of top correlations to print. Default: 10
        """
        if self.results is None:
            raise RuntimeError("Must run experiment before printing summary.")

        if self.results.distances_df is None:
            self.compute_distances()

        print(f"\nTop {top_n} distances with {self.config.correlation_target}:")
        print("-" * 55)

        for _, row in self.results.distances_df.head(top_n).iterrows():
            p_value = row["p_value"]
            if pd.isna(p_value):
                sig = ""
                p_str = ""
            else:
                sig = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                p_str = f" (p={p_value:.3f})"
            print(
                f"  {row['complexity_metric']:25s}: dist={row['distance']:+.3f}"
                f"{p_str} {sig}"
            )

        if self.results.per_classifier_distances_df is not None:
            df = self.results.per_classifier_distances_df
            for ml_metric in df["ml_metric"].unique():
                subset = df[df["ml_metric"] == ml_metric].head(top_n)
                print(f"\nTop {top_n} per-classifier distances ({ml_metric}):")
                print("-" * 60)
                for _, row in subset.iterrows():
                    print(
                        f"  {row['complexity_metric']:25s}: "
                        f"mean_dist={row['mean_distance']:+.3f} "
                        f"± {row['std_distance']:.3f}"
                    )
