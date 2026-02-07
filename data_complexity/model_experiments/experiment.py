"""
Generic experiment framework for complexity vs ML performance analysis.

Provides a configurable, reusable framework for running experiments that
measure the correlation between data complexity metrics and ML classifier
performance.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from data_complexity.metrics import complexity_metrics
from data_complexity.model_experiments.ml import (
    AbstractMLModel,
    get_default_models,
    evaluate_models,
    get_best_metric,
    get_mean_metric,
    get_metrics_dict,
    get_metrics_from_names,
)
from data_complexity.model_experiments.plotting import (
    plot_correlations,
    plot_metric_vs_accuracy,
    plot_summary,
    plot_correlation_heatmap,
    plot_model_comparison,
)


class PlotType(Enum):
    """Types of plots that can be generated."""

    CORRELATIONS = auto()
    METRIC_VS_ACCURACY = auto()
    SUMMARY = auto()
    HEATMAP = auto()


@dataclass
class ParameterSpec:
    """
    Specification for a parameter to vary in an experiment.

    Parameters
    ----------
    name : str
        Parameter name as passed to get_dataset().
    values : list
        Values to iterate over.
    label_format : str
        Format string for labeling data points. Use {name} and {value} placeholders.
    """

    name: str
    values: List[Any]
    label_format: str = "{name}={value}"

    def format_label(self, value: Any) -> str:
        """Format a label for a specific parameter value."""
        return self.label_format.format(name=self.name, value=value)


@dataclass
class DatasetSpec:
    """
    Specification for the dataset type and fixed parameters.

    Parameters
    ----------
    dataset_type : str
        Type of dataset ('Gaussian', 'Moons', 'Circles', 'Blobs', 'XOR').
    fixed_params : dict
        Parameters that remain constant across experiment iterations.
    num_samples : int
        Number of samples per dataset. Default: 400
    """

    dataset_type: str
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    num_samples: int = 400


@dataclass
class ExperimentConfig:
    """
    Configuration for a complexity vs ML experiment.

    Parameters
    ----------
    dataset : DatasetSpec
        Dataset specification.
    vary_parameter : ParameterSpec
        Parameter to vary across experiment iterations.
    models : list of AbstractMLModel, optional
        ML models to evaluate. Default: get_default_models()
    ml_metrics : list of str
        ML metrics to compute. Default: ['accuracy', 'f1']
    cv_folds : int
        Cross-validation folds. Default: 5
    name : str, optional
        Experiment name. Auto-generated if None.
    save_dir : Path, optional
        Directory to save results. Default: results/{name}/
    plots : list of PlotType
        Plot types to generate. Default: [CORRELATIONS, SUMMARY]
    correlation_target : str
        ML metric to correlate against. Default: 'best_accuracy'
    """

    dataset: DatasetSpec
    vary_parameter: ParameterSpec
    models: Optional[List[AbstractMLModel]] = None
    ml_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1"])
    cv_folds: int = 5
    name: Optional[str] = None
    save_dir: Optional[Path] = None
    plots: List[PlotType] = field(
        default_factory=lambda: [PlotType.CORRELATIONS, PlotType.SUMMARY]
    )
    correlation_target: str = "best_accuracy"

    def __post_init__(self):
        """Generate name and save_dir if not provided."""
        if self.name is None:
            self.name = self._generate_name()
        if self.save_dir is None:
            self.save_dir = Path(__file__).parent / "runs" / "results" / self.name

    def _generate_name(self) -> str:
        """Generate experiment name from dataset type and varied parameter."""
        return f"{self.dataset.dataset_type.lower()}_{self.vary_parameter.name}"


class ExperimentResults:
    """
    Container for experiment results with DataFrame storage.

    Stores complexity metrics and ML performance for each parameter value,
    and provides methods for correlation analysis.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._complexity_rows: List[Dict[str, Any]] = []
        self._ml_rows: List[Dict[str, Any]] = []
        self._complexity_df: Optional[pd.DataFrame] = None
        self._ml_df: Optional[pd.DataFrame] = None
        self._correlations_df: Optional[pd.DataFrame] = None

    def add_result(
        self,
        param_value: Any,
        complexity_metrics_dict: Dict[str, float],
        ml_results: Dict[str, Dict[str, Dict[str, float]]],
    ) -> None:
        """
        Add results for a single parameter value.

        Parameters
        ----------
        param_value : Any
            The parameter value used for this iteration.
        complexity_metrics_dict : dict
            Complexity metric name -> value.
        ml_results : dict
            Model name -> metric name -> {'mean': float, 'std': float}
        """
        complexity_row = {
            "param_value": param_value,
            "param_label": self.config.vary_parameter.format_label(param_value),
        }
        complexity_row.update(complexity_metrics_dict)
        self._complexity_rows.append(complexity_row)

        ml_row = {"param_value": param_value}

        for metric in self.config.ml_metrics:
            ml_row[f"best_{metric}"] = get_best_metric(ml_results, metric)
            ml_row[f"mean_{metric}"] = get_mean_metric(ml_results, metric)

        for model_name, metrics in ml_results.items():
            for metric in self.config.ml_metrics:
                if metric in metrics:
                    ml_row[f"{model_name}_{metric}"] = metrics[metric]["mean"]

        self._ml_rows.append(ml_row)

    def finalize(self) -> None:
        """Convert collected rows to DataFrames."""
        self._complexity_df = pd.DataFrame(self._complexity_rows)
        self._ml_df = pd.DataFrame(self._ml_rows)

    @property
    def complexity_df(self) -> pd.DataFrame:
        """Get complexity metrics DataFrame."""
        if self._complexity_df is None:
            self.finalize()
        return self._complexity_df

    @property
    def ml_df(self) -> pd.DataFrame:
        """Get ML performance DataFrame."""
        if self._ml_df is None:
            self.finalize()
        return self._ml_df

    @property
    def correlations_df(self) -> Optional[pd.DataFrame]:
        """Get correlations DataFrame (computed separately)."""
        return self._correlations_df

    @correlations_df.setter
    def correlations_df(self, df: pd.DataFrame) -> None:
        """Set correlations DataFrame."""
        self._correlations_df = df

    def get_param_values(self) -> List[Any]:
        """Get list of parameter values."""
        return self.complexity_df["param_value"].tolist()


class Experiment:
    """
    Main experiment runner for complexity vs ML performance analysis.

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
        Execute the experiment loop.

        Parameters
        ----------
        verbose : bool
            Print progress. Default: True

        Returns
        -------
        ExperimentResults
            Results container with complexity and ML DataFrames.
        """
        self._load_dataset_loader()
        self.results = ExperimentResults(self.config)

        models = self.config.models or get_default_models()

        if verbose:
            print(f"Running experiment: {self.config.name}")
            print(f"  Dataset: {self.config.dataset.dataset_type}")
            print(f"  Varying: {self.config.vary_parameter.name}")
            print(f"  Values: {self.config.vary_parameter.values}")
            print()

        for param_value in self.config.vary_parameter.values:
            params = dict(self.config.dataset.fixed_params)
            params[self.config.vary_parameter.name] = param_value
            params["num_samples"] = self.config.dataset.num_samples
            params["name"] = self.config.vary_parameter.format_label(param_value)

            dataset = self._get_dataset(self.config.dataset.dataset_type, **params)
            self.datasets[param_value] = dataset  # Store for later visualization
            data = dataset.get_data_dict()
            X, y = data["X"], data["y"]

            complexity = complexity_metrics(dataset=data)
            complexity_dict = complexity.get_all_metrics_scalar()

            # Convert metric names from config to metric objects
            metrics = get_metrics_from_names(self.config.ml_metrics)
            ml_results = evaluate_models(
                data, models=models, metrics=metrics, cv_folds=self.config.cv_folds
            )

            self.results.add_result(param_value, complexity_dict, ml_results)

            if verbose:
                best_acc = get_best_metric(ml_results, "accuracy")
                print(f"  {params['name']}: best_accuracy={best_acc:.3f}")

        self.results.finalize()
        return self.results

    def compute_correlations(
        self, ml_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute correlations between complexity metrics and ML performance.

        Parameters
        ----------
        ml_column : str, optional
            ML metric column to correlate against.
            Default: config.correlation_target

        Returns
        -------
        pd.DataFrame
            Correlation results sorted by absolute correlation.
        """
        if self.results is None:
            raise RuntimeError("Must run experiment before computing correlations.")

        ml_column = ml_column or self.config.correlation_target
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
        Save results to CSVs and plots to PNGs.

        Parameters
        ----------
        save_dir : Path, optional
            Directory to save to. Default: config.save_dir
        """
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

        # Save CSVs to data/ subfolder
        self.results.complexity_df.to_csv(data_dir / "complexity_metrics.csv", index=False)
        self.results.ml_df.to_csv(data_dir / "ml_performance.csv", index=False)

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

        self.datasets = {}  # Clear since we don't have loaders for loaded results

        return self.results
