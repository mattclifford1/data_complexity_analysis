"""
Experiment class for complexity vs ML performance analysis.

The actual run, plot, correlation, and I/O logic lives in the sibling modules:
- runner.py       — experiment loop and parallel worker
- plotting.py     — visualization generation
- correlations.py — correlation computation
- io.py           — save/load to/from disk
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from data_complexity.experiments.pipeline import (
    runner,
    plotting as _plotting,
    correlations as _correlations,
    io,
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

    def compute_correlations(
        self,
        ml_column: Optional[str] = None,
        complexity_source: str = "train",
        ml_source: str = "test",
    ) -> pd.DataFrame:
        """
        Compute correlations between complexity metrics and ML performance.

        Parameters
        ----------
        ml_column : str, optional
            ML metric column to correlate against. Default: config.correlation_target
        complexity_source : str
            Which complexity to use: 'train' or 'test'. Default: 'train'
        ml_source : str
            Which ML results to use: 'train' or 'test'. Default: 'test'

        Returns
        -------
        pd.DataFrame
            Correlation results sorted by absolute correlation.
        """
        return _correlations.compute_correlations(
            self, ml_column=ml_column, complexity_source=complexity_source, ml_source=ml_source
        )

    def compute_complexity_correlations(self, source: str = "train") -> pd.DataFrame:
        """
        Compute pairwise Pearson correlations between complexity metrics.

        Always computes correlations for both train and test data (when available).
        Train correlations are stored in ``results.complexity_correlations_df`` (backward compat);
        test correlations are stored in ``results.complexity_correlations_test_df``.

        Parameters
        ----------
        source : str
            Which correlation matrix to return: 'train' or 'test'. Default: 'train'

        Returns
        -------
        pd.DataFrame
            N×N symmetric correlation matrix for the requested source.
        """
        return _correlations.compute_complexity_correlations(self, source=source)

    def compute_ml_correlations(self, source: str = "test") -> pd.DataFrame:
        """
        Compute pairwise Pearson correlations between ML performance metrics.

        Parameters
        ----------
        source : str
            Which ML data to use: 'train' or 'test'. Default: 'test'

        Returns
        -------
        pd.DataFrame
            N×N symmetric correlation matrix indexed and columned by metric names.
        """
        return _correlations.compute_ml_correlations(self, source=source)

    def compute_all_correlations(self) -> pd.DataFrame:
        """
        Compute correlations for all ML metric columns.

        Returns
        -------
        pd.DataFrame
            All correlations combined.
        """
        return _correlations.compute_all_correlations(self)

    def compute_per_classifier_correlations(
        self,
        complexity_source: str = "train",
        ml_source: str = "test",
    ) -> pd.DataFrame:
        """
        Compute complexity-vs-ML correlations aggregated per metric across classifiers.

        For each ML metric type (e.g., 'accuracy', 'f1'), computes the Pearson
        correlation between each complexity metric and every individual classifier's
        column, then reports mean and std of those correlations.

        Parameters
        ----------
        complexity_source : str
            'train' or 'test'. Default: 'train'
        ml_source : str
            'train' or 'test'. Default: 'test'

        Returns
        -------
        pd.DataFrame
            Columns: complexity_metric, ml_metric, mean_correlation,
                     std_correlation, abs_mean_correlation.
            Sorted by abs_mean_correlation descending.
        """
        return _correlations.compute_per_classifier_correlations(
            self, complexity_source=complexity_source, ml_source=ml_source
        )

    def _compute_single_correlation(self, ml_column: str) -> pd.DataFrame:
        """Compute correlations for a single ML column."""
        return _correlations._compute_single_correlation(self, ml_column)

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
            'complexity_correlations_train' and 'complexity_correlations_test').
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

        if self.results.correlations_df is None:
            self.compute_correlations()

        print(f"\nTop {top_n} correlations with {self.config.correlation_target}:")
        print("-" * 55)

        for _, row in self.results.correlations_df.head(top_n).iterrows():
            sig = "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(
                f"  {row['complexity_metric']:25s}: r={row['correlation']:+.3f} "
                f"(p={row['p_value']:.3f}) {sig}"
            )

        if self.results.per_classifier_correlations_df is not None:
            df = self.results.per_classifier_correlations_df
            for ml_metric in df["ml_metric"].unique():
                subset = df[df["ml_metric"] == ml_metric].head(top_n)
                print(f"\nTop {top_n} per-classifier correlations ({ml_metric}):")
                print("-" * 60)
                for _, row in subset.iterrows():
                    print(
                        f"  {row['complexity_metric']:25s}: "
                        f"mean_r={row['mean_correlation']:+.3f} "
                        f"± {row['std_correlation']:.3f}"
                    )
