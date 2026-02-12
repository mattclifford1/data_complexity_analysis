"""
Generic experiment framework for complexity vs ML performance analysis.

Provides a configurable, reusable framework for running experiments that
measure the correlation between data complexity metrics and ML classifier
performance. All experiments use train/test splits: complexity is computed
on both splits independently, and ML models are trained on the training set
and evaluated on both.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from data_complexity.metrics import complexity_metrics
from data_complexity.model_experiments.ml import (
    AbstractMLModel,
    get_default_models,
    evaluate_models_train_test,
    get_best_metric,
    get_mean_metric,
    get_metrics_from_names,
)
from data_complexity.model_experiments.plotting import (
    plot_correlations,
    plot_metric_vs_accuracy,
    plot_summary,
    plot_correlation_heatmap,
    plot_model_comparison,
)

from data_complexity.model_experiments.experiment_utils import (
    DatasetSpec,
    ExperimentConfig,
    ExperimentResults,
    ParameterSpec,
    _average_dicts,
    _average_ml_results,
    PlotType,
)



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
        self.results: Optional[ExperimentResults] = None
        self._get_dataset = None
        self.datasets: Dict[Any, Any] = {}  # Store loaders for visualization

    def _load_dataset_loader(self):
        """Lazy load data_loaders to avoid import at module level."""
        if self._get_dataset is None:
            import data_loaders
            from data_loaders import get_dataset

            self._get_dataset = get_dataset

    def run(self, verbose: bool = True) -> ExperimentResults:
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

        Returns
        -------
        ExperimentResults
            Results container with train/test complexity and ML DataFrames.
        """
        self._load_dataset_loader()

        self.results = ExperimentResults(self.config)

        models = self.config.models or get_default_models()
        metrics = get_metrics_from_names(self.config.ml_metrics)
        train_size = self.config.dataset.train_size
        cv_folds = self.config.cv_folds

        if verbose:
            print(f"Running experiment: {self.config.name}")
            print(f"  Dataset: {self.config.dataset.dataset_type}")
            print(f"  Varying: {self.config.vary_parameter.name}")
            print(f"  Values: {self.config.vary_parameter.values}")
            print(f"  Train size: {train_size}, Seeds: {cv_folds}")
            print()

        for param_value in self.config.vary_parameter.values:
            # Build dataset params (include all parameters)
            params = dict(self.config.dataset.fixed_params)
            params[self.config.vary_parameter.name] = param_value
            params["num_samples"] = self.config.dataset.num_samples
            params["name"] = self.config.vary_parameter.format_label(param_value)

            dataset = self._get_dataset(self.config.dataset.dataset_type, **params)
            self.datasets[param_value] = dataset

            # Determine minority_reduce_scaler
            minority_reduce_scaler = self._get_minority_reduce_scaler(param_value)

            # Accumulate results across seeds
            train_complexity_accum: List[Dict[str, float]] = []
            test_complexity_accum: List[Dict[str, float]] = []
            train_ml_accum: List[Dict] = []
            test_ml_accum: List[Dict] = []

            for seed_i in range(cv_folds):
                # Use dataset's built-in proportional_split with minority_reduce_scaler
                train_data, test_data = dataset.get_train_test_split(
                             train_size=train_size,
                             minority_reduce_scaler=minority_reduce_scaler,
                             minority_reduce_scaler_test=None,
                             equal_test=None,
                             seed=42 + seed_i)

                # Complexity on train and test
                train_cmplx = complexity_metrics(dataset=train_data).get_all_metrics_scalar()
                test_cmplx = complexity_metrics(dataset=test_data).get_all_metrics_scalar()

                # ML: train on train, evaluate on both
                train_ml, test_ml = evaluate_models_train_test(
                    train_data, test_data, models=models, metrics=metrics,
                )

                train_complexity_accum.append(train_cmplx)
                test_complexity_accum.append(test_cmplx)
                train_ml_accum.append(train_ml)
                test_ml_accum.append(test_ml)

            # Average across seeds
            avg_train_complexity = _average_dicts(train_complexity_accum)
            avg_test_complexity = _average_dicts(test_complexity_accum)
            avg_train_ml = _average_ml_results(train_ml_accum)
            avg_test_ml = _average_ml_results(test_ml_accum)

            self.results.add_split_result(
                param_value, avg_train_complexity, avg_test_complexity,
                avg_train_ml, avg_test_ml,
            )

            if verbose:
                best_acc = get_best_metric(avg_test_ml, "accuracy")
                print(f"  {params['name']}: best_test_accuracy={best_acc:.3f}")

        self.results.finalize()
        return self.results

    def _get_minority_reduce_scaler(self, param_value: Any) -> Optional[float]:
        """Determine minority_reduce_scaler for the current iteration."""
        if self.config.vary_parameter.name == "minority_reduce_scaler":
            return param_value
        return self.config.dataset.fixed_params.get("minority_reduce_scaler")

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

        ml_column = ml_column or self.config.correlation_target
        complexity_df = self.results._get_complexity_df(complexity_source)
        ml_df = self.results._get_ml_df(ml_source)

        metric_cols = [
            c for c in complexity_df.columns if c not in ("param_value", "param_label")
        ]
        ml_values = ml_df[ml_column].values
        results = []

        for metric in metric_cols:
            values = complexity_df[metric].values

            if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
                continue

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

    def _compute_single_correlation(self, ml_column: str) -> pd.DataFrame:
        """Compute correlations for a single ML column."""
        complexity_df = self.results.complexity_df
        ml_df = self.results.ml_df

        metric_cols = [
            c for c in complexity_df.columns if c not in ("param_value", "param_label")
        ]
        ml_values = ml_df[ml_column].values
        results = []

        for metric in metric_cols:
            values = complexity_df[metric].values

            if np.std(values) == 0 or np.any(np.isnan(values)) or np.any(np.isnan(ml_values)):
                continue

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
    ) -> Dict[PlotType, "plt.Figure"]:
        """
        Generate experiment plots.

        Parameters
        ----------
        plot_types : list of PlotType, optional
            Plot types to generate. Default: config.plots

        Returns
        -------
        dict
            PlotType -> matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self.results is None:
            raise RuntimeError("Must run experiment before plotting.")

        plot_types = plot_types or self.config.plots
        figures = {}

        if self.results.correlations_df is None:
            self.compute_correlations()

        for pt in plot_types:
            if pt == PlotType.CORRELATIONS:
                fig = plot_correlations(
                    self.results.correlations_df,
                    title=f"{self.config.name}: Complexity vs {self.config.correlation_target}",
                )
                figures[pt] = fig

            elif pt == PlotType.SUMMARY:
                fig = plot_summary(
                    self.results.complexity_df,
                    self.results.ml_df,
                    self.results.correlations_df,
                    ml_column=self.config.correlation_target,
                    top_n=6,
                )
                figures[pt] = fig

            elif pt == PlotType.HEATMAP:
                all_corr = self.compute_all_correlations()
                fig = plot_model_comparison(all_corr)
                figures[pt] = fig

        return figures

    def save(self, save_dir: Optional[Path] = None) -> None:
        """
        Save results to CSVs, plots to PNGs, and experiment metadata to JSON.

        Parameters
        ----------
        save_dir : Path, optional
            Directory to save to. Default: config.save_dir
        """
        import json
        from datetime import datetime
        import matplotlib.pyplot as plt

        if self.results is None:
            raise RuntimeError("Must run experiment before saving.")

        save_dir = save_dir or self.config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create subfolders for organized results
        data_dir = save_dir / "data"
        plots_dir = save_dir / "plots"
        datasets_dir = save_dir / "datasets"

        data_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)
        datasets_dir.mkdir(exist_ok=True)

        # Save experiment metadata
        models = self.config.models or get_default_models()
        metadata = {
            "experiment_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "type": self.config.dataset.dataset_type,
                "num_samples": self.config.dataset.num_samples,
                "train_size": self.config.dataset.train_size,
                "fixed_params": self.config.dataset.fixed_params,
            },
            "vary_parameter": {
                "name": self.config.vary_parameter.name,
                "values": self.config.vary_parameter.values,
                "label_format": self.config.vary_parameter.label_format,
            },
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
            "ml_metrics": self.config.ml_metrics,
            "cv_folds": self.config.cv_folds,
            "correlation_target": self.config.correlation_target,
            "plots": [pt.name for pt in self.config.plots],
        }

        with open(save_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save backward-compat CSVs (train complexity + test ML)
        self.results.complexity_df.to_csv(data_dir / "complexity_metrics.csv", index=False)
        self.results.ml_df.to_csv(data_dir / "ml_performance.csv", index=False)

        # Save train/test split CSVs if available
        if self.results.train_complexity_df is not None:
            self.results.train_complexity_df.to_csv(
                data_dir / "train_complexity_metrics.csv", index=False
            )
        if self.results.test_complexity_df is not None:
            self.results.test_complexity_df.to_csv(
                data_dir / "test_complexity_metrics.csv", index=False
            )
        if self.results.train_ml_df is not None:
            self.results.train_ml_df.to_csv(
                data_dir / "train_ml_performance.csv", index=False
            )
        if self.results.test_ml_df is not None:
            self.results.test_ml_df.to_csv(
                data_dir / "test_ml_performance.csv", index=False
            )

        if self.results.correlations_df is not None:
            self.results.correlations_df.to_csv(data_dir / "correlations.csv", index=False)

        # Save plots to plots/ subfolder
        figures = self.plot()
        for plot_type, fig in figures.items():
            filename = f"{plot_type.name.lower()}.png"
            fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Save dataset visualizations to datasets/ subfolder
        # Create dual plot: full dataset + train/test split
        for param_value, dataset in self.datasets.items():
            param_label = self.config.vary_parameter.format_label(param_value)

            # Try to create train/test split plot
            # Some datasets may not support splitting (e.g., minority_reduce_scaler=1)
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                # Left: Full dataset
                dataset.plot_dataset(ax=axes[0])
                axes[0].set_title("Full Dataset", fontsize=12, fontweight='bold')

                # Middle & Right: Train/test split
                dataset.plot_train_test_split(ax=(axes[1], axes[2]))

                # Add overall title with parameter value
                fig.suptitle(f"Dataset: {param_label}", fontsize=14, fontweight='bold', y=1.02)

            except (ValueError, AttributeError):
                # Fallback: single plot if train/test split not supported
                plt.close(fig)  # Close the 3-panel figure
                fig, ax = plt.subplots(figsize=(8, 6))
                dataset.plot_dataset(ax=ax)
                ax.set_title(f"Full Dataset: {param_label}", fontsize=12, fontweight='bold')

            # Sanitize label for filename (replace = with _)
            safe_label = param_label.replace("=", "_").replace(" ", "_")
            filename = f"dataset_{safe_label}.png"
            fig.savefig(datasets_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved results to: {save_dir}")
        print(f"  - Metadata: experiment_metadata.json")
        print(f"  - Data CSVs: data/")
        print(f"  - Plots: plots/")
        print(f"  - Datasets: datasets/")

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

    def _resolve_path(self, save_dir: Path, filename: str, subfolder: Optional[str] = None) -> Path:
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

    def load_results(self, save_dir: Optional[Path] = None) -> ExperimentResults:
        """
        Load previously saved results from CSVs.

        Parameters
        ----------
        save_dir : Path, optional
            Directory to load from. Default: config.save_dir

        Returns
        -------
        ExperimentResults
            Loaded results.
        """
        save_dir = save_dir or self.config.save_dir

        # Resolve paths with backwards compatibility for flat structure
        complexity_path = self._resolve_path(save_dir, "complexity_metrics.csv", "data")
        ml_path = self._resolve_path(save_dir, "ml_performance.csv", "data")
        corr_path = self._resolve_path(save_dir, "correlations.csv", "data")

        self.results = ExperimentResults(self.config)
        self.results._complexity_df = pd.read_csv(complexity_path)
        self.results._ml_df = pd.read_csv(ml_path)

        if corr_path.exists():
            self.results._correlations_df = pd.read_csv(corr_path)

        # Load train/test CSVs if present
        train_complexity_path = self._resolve_path(
            save_dir, "train_complexity_metrics.csv", "data"
        )
        if train_complexity_path.exists():
            self.results._train_complexity_df = pd.read_csv(train_complexity_path)

        test_complexity_path = self._resolve_path(
            save_dir, "test_complexity_metrics.csv", "data"
        )
        if test_complexity_path.exists():
            self.results._test_complexity_df = pd.read_csv(test_complexity_path)

        train_ml_path = self._resolve_path(
            save_dir, "train_ml_performance.csv", "data"
        )
        if train_ml_path.exists():
            self.results._train_ml_df = pd.read_csv(train_ml_path)

        test_ml_path = self._resolve_path(
            save_dir, "test_ml_performance.csv", "data"
        )
        if test_ml_path.exists():
            self.results._test_ml_df = pd.read_csv(test_ml_path)

        self.datasets = {}  # Clear since we don't have loaders for loaded results

        return self.results
