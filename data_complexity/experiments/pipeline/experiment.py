"""
Experiment class for complexity vs ML performance analysis.

The actual run, plot, and I/O logic lives in the sibling modules:
- runner.py   — experiment loop and parallel worker
- plotting.py — visualization generation
- io.py       — save/load to/from disk
"""
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

from data_complexity.experiments.pipeline import runner, plotting as _plotting, io
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
            ML metric column to correlate against.
            Default: config.correlation_target
        complexity_source : str
            Which complexity to use: 'train' or 'test'. Default: 'train'
        ml_source : str
            Which ML results to use: 'train' or 'test'. Default: 'test'

        Returns
        -------
        pd.DataFrame
            Correlation results sorted by absolute correlation.
        """
        if self.results is None:
            raise RuntimeError("Must run experiment before computing correlations.")

        run_mode = self.config.run_mode
        if run_mode == RunMode.COMPLEXITY_ONLY:
            raise RuntimeError(
                "Cannot compute correlations with run_mode=COMPLEXITY_ONLY (no ML results)."
            )
        if run_mode == RunMode.ML_ONLY:
            raise RuntimeError(
                "Cannot compute correlations with run_mode=ML_ONLY (no complexity results)."
            )

        ml_column = ml_column or self.config.correlation_target
        complexity_df = self.results._get_complexity_df(complexity_source)
        ml_df = self.results._get_ml_df(ml_source)

        metric_cols = [
            c for c in complexity_df.columns if c not in ("param_value", "param_label")
        ]
        ml_values = ml_df[ml_column].values
        results = []

        if np.std(ml_values) == 0:
            correlations_df = pd.DataFrame(
                columns=["complexity_metric", "ml_metric", "correlation", "p_value", "abs_correlation"]
            )
            self.results.correlations_df = correlations_df
            return correlations_df

        for metric in metric_cols:
            values = complexity_df[metric].values

            if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (NearConstantInputWarning, ConstantInputWarning))
                r, p = stats.pearsonr(values, ml_values)
            results.append(
                {
                    "complexity_metric": metric,
                    "ml_metric": ml_column,
                    "correlation": r,
                    "p_value": p,
                    "abs_correlation": abs(r),
                }
            )

        correlations_df = pd.DataFrame(results).sort_values(
            "abs_correlation", ascending=False
        )
        self.results.correlations_df = correlations_df
        return correlations_df

    def compute_complexity_correlations(
        self,
        source: str = "train",
    ) -> pd.DataFrame:
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
        if self.results is None:
            raise RuntimeError("Must run experiment before computing correlations.")

        def _compute_for_source(src: str) -> Optional[pd.DataFrame]:
            complexity_df = self.results._get_complexity_df(src)
            if complexity_df is None:
                return None
            metric_cols = [
                c for c in complexity_df.columns
                if c not in ("param_value", "param_label") and not c.endswith("_std")
            ]
            valid_cols = [c for c in metric_cols if complexity_df[c].std() > 0]
            if not valid_cols:
                return None
            return complexity_df[valid_cols].corr(method="pearson")

        train_corr = _compute_for_source("train")
        if train_corr is not None:
            self.results.complexity_correlations_df = train_corr

        test_corr = _compute_for_source("test")
        if test_corr is not None:
            self.results.complexity_correlations_test_df = test_corr

        return train_corr if source == "train" else test_corr

    def compute_ml_correlations(
        self,
        source: str = "test",
    ) -> pd.DataFrame:
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
        if self.results is None:
            raise RuntimeError("Must run experiment before computing correlations.")
        if self.config.run_mode == RunMode.COMPLEXITY_ONLY:
            raise RuntimeError(
                "Cannot compute ML correlations with run_mode=COMPLEXITY_ONLY (no ML results)."
            )

        ml_df = self.results._get_ml_df(source)
        metric_cols = [
            c for c in ml_df.columns
            if c != "param_value" and not c.endswith("_std")
        ]
        # Drop constant metrics (zero std) — they produce NaN correlations
        valid_cols = [c for c in metric_cols if ml_df[c].std() > 0]

        corr_matrix = ml_df[valid_cols].corr(method="pearson")
        self.results.ml_correlations_df = corr_matrix
        return corr_matrix

    def compute_all_correlations(self) -> pd.DataFrame:
        """
        Compute correlations for all ML metric columns.

        Returns
        -------
        pd.DataFrame
            All correlations combined.
        """
        if self.results is None:
            raise RuntimeError("Must run experiment before computing correlations.")

        ml_df = self.results.ml_df
        ml_cols = [c for c in ml_df.columns if c != "param_value"]

        all_corrs = []
        for ml_col in ml_cols:
            corr_df = self._compute_single_correlation(ml_col)
            all_corrs.append(corr_df)

        return pd.concat(all_corrs, ignore_index=True)

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
        if self.results is None:
            raise RuntimeError("Must run experiment before computing correlations.")

        complexity_df = self.results._get_complexity_df(complexity_source)
        ml_df = self.results._get_ml_df(ml_source)

        complexity_cols = [
            c for c in complexity_df.columns
            if c not in ("param_value", "param_label") and not c.endswith("_std")
        ]

        rows = []
        for metric_name in self.config.ml_metrics:
            # Per-classifier columns: end with _{metric_name}, no best_/mean_ prefix, no _std suffix
            classifier_cols = [
                c for c in ml_df.columns
                if c.endswith(f"_{metric_name}")
                and not c.startswith("best_")
                and not c.startswith("mean_")
                and not c.endswith("_std")
            ]

            for complexity_col in complexity_cols:
                comp_values = complexity_df[complexity_col].values
                if np.std(comp_values) == 0 or np.any(np.isnan(comp_values)):
                    continue

                rs = []
                for clf_col in classifier_cols:
                    ml_values = ml_df[clf_col].values
                    if np.std(ml_values) == 0 or np.any(np.isnan(ml_values)):
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "ignore", (NearConstantInputWarning, ConstantInputWarning)
                        )
                        r, _ = stats.pearsonr(comp_values, ml_values)
                    if not np.isnan(r):
                        rs.append(r)

                if not rs:
                    continue

                mean_r = float(np.mean(rs))
                rows.append(
                    {
                        "complexity_metric": complexity_col,
                        "ml_metric": metric_name,
                        "mean_correlation": mean_r,
                        "std_correlation": float(np.std(rs)),
                        "abs_mean_correlation": abs(mean_r),
                    }
                )

        df = pd.DataFrame(rows).sort_values("abs_mean_correlation", ascending=False)
        self.results.per_classifier_correlations_df = df
        return df

    def _compute_single_correlation(self, ml_column: str) -> pd.DataFrame:
        """Compute correlations for a single ML column."""
        complexity_df = self.results.complexity_df
        ml_df = self.results.ml_df

        metric_cols = [
            c for c in complexity_df.columns if c not in ("param_value", "param_label")
        ]
        ml_values = ml_df[ml_column].values
        results = []

        if np.std(ml_values) == 0:
            return pd.DataFrame(
                columns=["complexity_metric", "ml_metric", "correlation", "p_value", "abs_correlation"]
            )

        for metric in metric_cols:
            values = complexity_df[metric].values

            if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (NearConstantInputWarning, ConstantInputWarning))
                r, p = stats.pearsonr(values, ml_values)
            results.append(
                {
                    "complexity_metric": metric,
                    "ml_metric": ml_column,
                    "correlation": r,
                    "p_value": p,
                    "abs_correlation": abs(r),
                }
            )

        if not results:
            raise ValueError("No valid correlations computed.")

        return pd.DataFrame(results).sort_values("abs_correlation", ascending=False)

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
