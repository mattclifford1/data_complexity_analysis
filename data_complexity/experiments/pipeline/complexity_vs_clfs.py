"""
Generic experiment framework for complexity vs ML performance analysis.

Provides a configurable, reusable framework for running experiments that
measure the correlation between data complexity metrics and ML classifier
performance. All experiments use train/test splits: complexity is computed
on both splits independently, and ML models are trained on the training set
and evaluated on both.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning, NearConstantInputWarning
from tqdm import tqdm

from data_complexity.metrics import ComplexityMetrics
from data_complexity.experiments.classification import (
    get_default_models,
    evaluate_models_train_test,
    get_best_metric,
    get_metrics_from_names,
)
from data_complexity.experiments.plotting import (
    plot_correlations,
    plot_summary,
    plot_model_comparison,
    plot_metrics_vs_parameter,
    plot_models_vs_parameter,
    plot_complexity_metrics_vs_parameter,
    plot_models_vs_parameter_combined,
    plot_complexity_metrics_vs_parameter_combined,
    plot_datasets_overview,
    plot_complexity_correlations_heatmap,
)

from data_complexity.experiments.pipeline.utils import (
    DatasetSpec as DatasetSpec,
    ExperimentConfig,
    ExperimentResultsContainer,
    ParameterSpec as ParameterSpec,
    _average_dicts,
    _average_ml_results,
    _std_dicts,
    PlotType,
    RunMode,
    make_json_safe_dict,
    make_json_safe_list,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _run_dataset_spec_worker(
    dataset_spec: "DatasetSpec",
    config: "ExperimentConfig",
    models: List,
    metrics: List,
) -> Dict[str, Any]:
    """Worker executed in a subprocess for one DatasetSpec.

    Parameters
    ----------
    dataset_spec : DatasetSpec
        The dataset specification for this worker.
    config : ExperimentConfig
        Experiment configuration (must be picklable — no callable fields).
    models : list
        ML model instances to evaluate.
    metrics : list
        ML metric instances to compute.

    Returns
    -------
    dict
        Averaged results across all seeds for this dataset spec.
    """
    from data_loaders import get_dataset  # type: ignore

    dataset = get_dataset(
        dataset_name=dataset_spec.dataset_type,
        name=dataset_spec.label,
        **dataset_spec.fixed_params,
    )

    run_mode = config.run_mode
    train_complexity_accum: List[Dict[str, float]] = []
    test_complexity_accum: List[Dict[str, float]] = []
    train_ml_accum: List[Dict] = []
    test_ml_accum: List[Dict] = []

    for seed_i in range(config.cv_folds):
        train_data, test_data = dataset.get_train_test_split(seed=42 + seed_i)

        if run_mode != RunMode.ML_ONLY:
            train_cmplx = ComplexityMetrics(dataset=train_data).get_all_metrics_scalar()
            test_cmplx = ComplexityMetrics(dataset=test_data).get_all_metrics_scalar()
            train_complexity_accum.append(train_cmplx)
            test_complexity_accum.append(test_cmplx)

        if run_mode != RunMode.COMPLEXITY_ONLY:
            train_ml, test_ml = evaluate_models_train_test(
                train_data, test_data, models=models, metrics=metrics,
            )
            train_ml_accum.append(train_ml)
            test_ml_accum.append(test_ml)

    return {
        "label": dataset_spec.label,
        "avg_train_complexity": _average_dicts(train_complexity_accum) if train_complexity_accum else None,
        "avg_test_complexity": _average_dicts(test_complexity_accum) if test_complexity_accum else None,
        "std_train_complexity": _std_dicts(train_complexity_accum) if train_complexity_accum else None,
        "std_test_complexity": _std_dicts(test_complexity_accum) if test_complexity_accum else None,
        "avg_train_ml": _average_ml_results(train_ml_accum) if train_ml_accum else None,
        "avg_test_ml": _average_ml_results(test_ml_accum) if test_ml_accum else None,
    }


# Keep old name as alias for backwards compatibility
_run_param_value_worker = _run_dataset_spec_worker


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
        self._load_dataset_loader()

        self.results = ExperimentResultsContainer(self.config)

        models = self.config.models or get_default_models()
        metrics = get_metrics_from_names(self.config.ml_metrics)
        cv_folds = self.config.cv_folds

        if verbose:
            print(f"Running experiment: {self.config.name}")
            print(f"  Datasets ({len(self.config.datasets)}):")
            for spec in self.config.datasets:
                print(f"    {spec.label}: {spec.dataset_type} {spec.fixed_params}")
            print(f"  x_label: {self.config.x_label}")
            print(f"  Seeds: {cv_folds}")
            if n_jobs != 1:
                print(f"  Parallel workers: {n_jobs if n_jobs > 0 else 'auto'}")
            print()

        if n_jobs == 1:
            # --- sequential loop ---
            for dataset_spec in tqdm(self.config.datasets, desc="Datasets"):
                # Build the dataset (train/test split and postprocessing handled by dataset object)
                dataset = self._get_dataset(
                    dataset_name=dataset_spec.dataset_type,
                    name=dataset_spec.label,
                    **dataset_spec.fixed_params,
                )
                # Store dataset for visualization later (keyed by label string)
                self.datasets[dataset_spec.label] = dataset

                # Accumulate results across seeds
                run_mode = self.config.run_mode
                train_complexity_accum: List[Dict[str, float]] = []
                test_complexity_accum: List[Dict[str, float]] = []
                train_ml_accum: List[Dict] = []
                test_ml_accum: List[Dict] = []

                for seed_i in tqdm(range(cv_folds), desc=f"Dataset {dataset_spec.label}", leave=False):
                    # Use dataset's built-in proportional_split with all params already set
                    train_data, test_data = dataset.get_train_test_split(seed=42 + seed_i)

                    if run_mode != RunMode.ML_ONLY:
                        train_cmplx = ComplexityMetrics(dataset=train_data).get_all_metrics_scalar()
                        test_cmplx = ComplexityMetrics(dataset=test_data).get_all_metrics_scalar()
                        train_complexity_accum.append(train_cmplx)
                        test_complexity_accum.append(test_cmplx)

                    if run_mode != RunMode.COMPLEXITY_ONLY:
                        train_ml, test_ml = evaluate_models_train_test(
                            train_data, test_data, models=models, metrics=metrics,
                        )
                        train_ml_accum.append(train_ml)
                        test_ml_accum.append(test_ml)

                # Average and std across seeds
                avg_train_complexity = _average_dicts(train_complexity_accum) if train_complexity_accum else None
                avg_test_complexity = _average_dicts(test_complexity_accum) if test_complexity_accum else None
                std_train_complexity = _std_dicts(train_complexity_accum) if train_complexity_accum else None
                std_test_complexity = _std_dicts(test_complexity_accum) if test_complexity_accum else None
                avg_train_ml = _average_ml_results(train_ml_accum) if train_ml_accum else None
                avg_test_ml = _average_ml_results(test_ml_accum) if test_ml_accum else None

                self.results.add_split_result(
                    param_value=dataset_spec.label,
                    train_complexity_dict=avg_train_complexity,
                    test_complexity_dict=avg_test_complexity,
                    train_ml_results=avg_train_ml,
                    test_ml_results=avg_test_ml,
                    train_complexity_std_dict=std_train_complexity,
                    test_complexity_std_dict=std_test_complexity,
                )

                if verbose and run_mode != RunMode.COMPLEXITY_ONLY and avg_test_ml:
                    best_acc = get_best_metric(avg_test_ml, "accuracy")
                    print(f"  {dataset_spec.label}: best_test_accuracy={best_acc:.3f}")

        else:
            # --- parallel branch ---
            from concurrent.futures import ProcessPoolExecutor, as_completed

            max_workers = n_jobs if n_jobs > 0 else None
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                for dataset_spec in self.config.datasets:
                    future = pool.submit(
                        _run_dataset_spec_worker,
                        dataset_spec, self.config, models, metrics,
                    )
                    futures[future] = dataset_spec

                results_by_label: Dict[str, Dict[str, Any]] = {}
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Datasets (parallel)",
                ):
                    result = future.result()  # re-raises worker exceptions in main process
                    results_by_label[result["label"]] = result

            # Restore original order then store
            for dataset_spec in self.config.datasets:
                r = results_by_label[dataset_spec.label]
                self.results.add_split_result(
                    param_value=dataset_spec.label,
                    train_complexity_dict=r["avg_train_complexity"],
                    test_complexity_dict=r["avg_test_complexity"],
                    train_ml_results=r["avg_train_ml"],
                    test_ml_results=r["avg_test_ml"],
                    train_complexity_std_dict=r["std_train_complexity"],
                    test_complexity_std_dict=r["std_test_complexity"],
                )
                if verbose and self.config.run_mode != RunMode.COMPLEXITY_ONLY and r["avg_test_ml"]:
                    best_acc = get_best_metric(r["avg_test_ml"], "accuracy")
                    print(f"  {dataset_spec.label}: best_test_accuracy={best_acc:.3f}")

            # Regenerate datasets in the main process for visualisation.
            # Dataset construction is cheap; we only need the objects for plotting.
            for dataset_spec in self.config.datasets:
                self.datasets[dataset_spec.label] = self._get_dataset(
                    dataset_name=dataset_spec.dataset_type,
                    name=dataset_spec.label,
                    **dataset_spec.fixed_params,
                )

        # Convert accumulated results to DataFrames
        self.results.covert_to_df()
        return self.results

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

        Parameters
        ----------
        source : str
            Which complexity data to use: 'train' or 'test'. Default: 'train'

        Returns
        -------
        pd.DataFrame
            N×N symmetric correlation matrix indexed and columned by metric names.
        """
        if self.results is None:
            raise RuntimeError("Must run experiment before computing correlations.")

        complexity_df = self.results._get_complexity_df(source)
        metric_cols = [
            c for c in complexity_df.columns
            if c not in ("param_value", "param_label") and not c.endswith("_std")
        ]
        # Drop constant metrics (zero std) — they produce NaN correlations
        valid_cols = [c for c in metric_cols if complexity_df[c].std() > 0]

        corr_matrix = complexity_df[valid_cols].corr(method="pearson")
        self.results.complexity_correlations_df = corr_matrix
        return corr_matrix

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
        if self.results is None:
            raise RuntimeError("Must run experiment before plotting.")

        plot_types = plot_types or self.config.plots
        figures = {}
        run_mode = self.config.run_mode

        if run_mode == RunMode.BOTH and self.results.correlations_df is None:
            self.compute_correlations()

        if PlotType.COMPLEXITY_CORRELATIONS in plot_types and self.results.complexity_correlations_df is None:
            self.compute_complexity_correlations()

        if PlotType.ML_CORRELATIONS in plot_types and self.results.ml_correlations_df is None:
            if self.config.run_mode != RunMode.COMPLEXITY_ONLY:
                self.compute_ml_correlations()

        for pt in plot_types:
            if pt == PlotType.CORRELATIONS:
                if run_mode != RunMode.BOTH:
                    import warnings
                    warnings.warn(
                        f"Skipping {pt.name} plot: requires both complexity and ML results "
                        f"(run_mode={run_mode.value})."
                    )
                    continue
                fig = plot_correlations(
                    self.results.correlations_df,
                    title=f"{self.config.name}: Complexity vs {self.config.correlation_target}",
                )
                figures[pt] = fig

            elif pt == PlotType.SUMMARY:
                if run_mode != RunMode.BOTH:
                    import warnings
                    warnings.warn(
                        f"Skipping {pt.name} plot: requires both complexity and ML results "
                        f"(run_mode={run_mode.value})."
                    )
                    continue
                fig = plot_summary(
                    self.results.complexity_df,
                    self.results.ml_df,
                    self.results.correlations_df,
                    ml_column=self.config.correlation_target,
                    top_n=6,
                )
                figures[pt] = fig

            elif pt == PlotType.HEATMAP:
                if run_mode != RunMode.BOTH:
                    import warnings
                    warnings.warn(
                        f"Skipping {pt.name} plot: requires both complexity and ML results "
                        f"(run_mode={run_mode.value})."
                    )
                    continue
                all_corr = self.compute_all_correlations()
                fig = plot_model_comparison(all_corr)
                figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_TRAIN:
                if self.results.train_complexity_df is not None and self.results.train_ml_df is not None:
                    fig = plot_metrics_vs_parameter(
                        complexity_df=self.results.train_complexity_df,
                        ml_df=self.results.train_ml_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Train — metrics vs {self.config.x_label}",
                    )
                    figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_TEST:
                if self.results.test_complexity_df is not None and self.results.test_ml_df is not None:
                    fig = plot_metrics_vs_parameter(
                        complexity_df=self.results.test_complexity_df,
                        ml_df=self.results.test_ml_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Test — metrics vs {self.config.x_label}",
                    )
                    figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_MODELS_TRAIN:
                if self.results.train_ml_df is not None:
                    fig = plot_models_vs_parameter(
                        ml_df=self.results.train_ml_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Models (Train) vs {self.config.x_label}",
                        ml_metrics=self.config.ml_metrics,
                    )
                    figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_MODELS_TEST:
                if self.results.test_ml_df is not None:
                    fig = plot_models_vs_parameter(
                        ml_df=self.results.test_ml_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Models (Test) vs {self.config.x_label}",
                        ml_metrics=self.config.ml_metrics,
                    )
                    figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_COMPLEXITY_TRAIN:
                if self.results.train_complexity_df is not None:
                    fig = plot_complexity_metrics_vs_parameter(
                        complexity_df=self.results.train_complexity_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Complexity (Train) vs {self.config.x_label}",
                    )
                    figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_COMPLEXITY_TEST:
                if self.results.test_complexity_df is not None:
                    fig = plot_complexity_metrics_vs_parameter(
                        complexity_df=self.results.test_complexity_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Complexity (Test) vs {self.config.x_label}",
                    )
                    figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_MODELS_COMBINED:
                if self.results.train_ml_df is not None and self.results.test_ml_df is not None:
                    fig = plot_models_vs_parameter_combined(
                        train_ml_df=self.results.train_ml_df,
                        test_ml_df=self.results.test_ml_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Models (Train vs Test) vs {self.config.x_label}",
                        ml_metrics=self.config.ml_metrics,
                    )
                    figures[pt] = fig

            elif pt == PlotType.LINE_PLOT_COMPLEXITY_COMBINED:
                if self.results.train_complexity_df is not None and self.results.test_complexity_df is not None:
                    fig = plot_complexity_metrics_vs_parameter_combined(
                        train_complexity_df=self.results.train_complexity_df,
                        test_complexity_df=self.results.test_complexity_df,
                        param_label_col="param_value",
                        x_label=self.config.x_label,
                        title=f"{self.config.name}: Complexity (Train vs Test) vs {self.config.x_label}",
                    )
                    figures[pt] = fig

            elif pt == PlotType.DATASETS_OVERVIEW:
                if self.datasets:
                    fig = plot_datasets_overview(
                        datasets=self.datasets,
                        format_label=lambda label: label,
                    )
                    figures[pt] = fig

            elif pt == PlotType.COMPLEXITY_CORRELATIONS:
                if self.results.complexity_correlations_df is None:
                    self.compute_complexity_correlations()
                fig = plot_complexity_correlations_heatmap(
                    self.results.complexity_correlations_df,
                    title=f"{self.config.name}: Complexity Metric Correlations",
                )
                figures[pt] = fig

            elif pt == PlotType.ML_CORRELATIONS:
                if self.config.run_mode == RunMode.COMPLEXITY_ONLY:
                    warnings.warn(
                        f"Skipping {pt.name} plot: requires ML results "
                        f"(run_mode={self.config.run_mode.value})."
                    )
                    continue
                if self.results.ml_correlations_df is None:
                    self.compute_ml_correlations()
                fig = plot_complexity_correlations_heatmap(
                    self.results.ml_correlations_df,
                    title=f"{self.config.name}: ML Metric Correlations",
                )
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
        import matplotlib.pyplot as plt

        if self.results is None:
            raise RuntimeError("Must run experiment before saving.")

        save_dir = save_dir or self.config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create subfolders for organized results
        data_dir = save_dir / "data"
        plots_dir = save_dir / f"plots-{self.config.name}"
        datasets_dir = save_dir / "datasets"

        data_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)
        datasets_dir.mkdir(exist_ok=True)

        # Save experiment metadata
        models = self.config.models or get_default_models()
        metadata = {
            "experiment_name": self.config.name,
            "datasets": [
                {
                    "label": spec.label,
                    "type": spec.dataset_type,
                    "params": make_json_safe_dict(spec.fixed_params),
                }
                for spec in self.config.datasets
            ],
            "x_label": self.config.x_label,
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
            "run_mode": self.config.run_mode.value,
        }

        with open(save_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save backward-compat CSVs (train complexity + test ML)
        if self.results.complexity_df is not None:
            self.results.complexity_df.to_csv(data_dir / "complexity_metrics.csv", index=False)
        if self.results.ml_df is not None:
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

        if self.results.complexity_correlations_df is not None:
            self.results.complexity_correlations_df.to_csv(
                data_dir / "complexity_correlations.csv"
            )

        if self.results.ml_correlations_df is not None:
            self.results.ml_correlations_df.to_csv(
                data_dir / "ml_correlations.csv"
            )

        if self.results.per_classifier_correlations_df is not None:
            self.results.per_classifier_correlations_df.to_csv(
                data_dir / "per_classifier_correlations.csv", index=False
            )

        # Save plots to plots/ subfolder
        figures = self.plot()
        for plot_type, fig in figures.items():
            filename = f"{plot_type.name.lower()}.png"
            fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Save dataset visualizations to datasets/ subfolder
        # Create dual plot: full dataset + train/test split
        for label, dataset in self.datasets.items():
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
        print(f"  - Plots: plots-{self.config.name}/")
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
        save_dir = save_dir or self.config.save_dir

        # Resolve paths with backwards compatibility for flat structure
        complexity_path = self._resolve_path(save_dir, "complexity_metrics.csv", "data")
        ml_path = self._resolve_path(save_dir, "ml_performance.csv", "data")
        corr_path = self._resolve_path(save_dir, "correlations.csv", "data")

        self.results = ExperimentResultsContainer(self.config)
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

        per_classifier_corr_path = self._resolve_path(
            save_dir, "per_classifier_correlations.csv", "data"
        )
        if per_classifier_corr_path.exists():
            self.results._per_classifier_correlations_df = pd.read_csv(per_classifier_corr_path)

        self.datasets = {}  # Clear since we don't have loaders for loaded results

        return self.results
