"""
Experiment runner: worker function and run() logic for the Experiment class.

This module contains the execution logic for running complexity vs ML experiments.
It is designed to be called from Experiment.run() via thin delegation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from tqdm import tqdm

from data_complexity.data_metrics.metrics import ComplexityMetrics
from data_complexity.experiments.classification import (
    get_default_models,
    evaluate_models_train_test,
    get_best_metric,
    get_metrics_from_names,
)
from data_complexity.experiments.pipeline.utils import (
    ExperimentResultsContainer,
    RunMode,
    _average_dicts,
    _average_ml_results,
    _std_dicts,
)

if TYPE_CHECKING:
    from data_complexity.experiments.pipeline.experiment import Experiment


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


def run(experiment: "Experiment", verbose: bool = True, n_jobs: int = 1) -> ExperimentResultsContainer:
    """
    Execute the experiment loop with train/test splits.

    For each parameter value, generates a dataset, then for each random
    seed (controlled by ``cv_folds``), splits into train/test using the
    dataset's built-in proportional_split(), computes complexity on both,
    and trains ML models on train to evaluate on both. Results are averaged
    across seeds.

    Parameters
    ----------
    experiment : Experiment
        The experiment instance whose config, results, and datasets are updated.
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
    experiment._load_dataset_loader()

    experiment.results = ExperimentResultsContainer(experiment.config)

    models = experiment.config.models or get_default_models()
    metrics = get_metrics_from_names(experiment.config.ml_metrics)
    cv_folds = experiment.config.cv_folds

    if verbose:
        print(f"Running experiment: {experiment.config.name}")
        print(f"  Datasets ({len(experiment.config.datasets)}):")
        for spec in experiment.config.datasets:
            print(f"    {spec.label}: {spec.dataset_type} {spec.fixed_params}")
        print(f"  x_label: {experiment.config.x_label}")
        print(f"  Seeds: {cv_folds}")
        if n_jobs != 1:
            print(f"  Parallel workers: {n_jobs if n_jobs > 0 else 'auto'}")
        print()

    if n_jobs == 1:
        # --- sequential loop ---
        for dataset_spec in tqdm(experiment.config.datasets, desc="Datasets"):
            # Build the dataset (train/test split and postprocessing handled by dataset object)
            dataset = experiment._get_dataset(
                dataset_name=dataset_spec.dataset_type,
                name=dataset_spec.label,
                **dataset_spec.fixed_params,
            )
            # Store dataset for visualization later (keyed by label string)
            experiment.datasets[dataset_spec.label] = dataset

            # Accumulate results across seeds
            run_mode = experiment.config.run_mode
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

            experiment.results.add_split_result(
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
            for dataset_spec in experiment.config.datasets:
                future = pool.submit(
                    _run_dataset_spec_worker,
                    dataset_spec, experiment.config, models, metrics,
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
        for dataset_spec in experiment.config.datasets:
            r = results_by_label[dataset_spec.label]
            experiment.results.add_split_result(
                param_value=dataset_spec.label,
                train_complexity_dict=r["avg_train_complexity"],
                test_complexity_dict=r["avg_test_complexity"],
                train_ml_results=r["avg_train_ml"],
                test_ml_results=r["avg_test_ml"],
                train_complexity_std_dict=r["std_train_complexity"],
                test_complexity_std_dict=r["std_test_complexity"],
            )
            if verbose and experiment.config.run_mode != RunMode.COMPLEXITY_ONLY and r["avg_test_ml"]:
                best_acc = get_best_metric(r["avg_test_ml"], "accuracy")
                print(f"  {dataset_spec.label}: best_test_accuracy={best_acc:.3f}")

        # Regenerate datasets in the main process for visualisation.
        # Dataset construction is cheap; we only need the objects for plotting.
        for dataset_spec in experiment.config.datasets:
            experiment.datasets[dataset_spec.label] = experiment._get_dataset(
                dataset_name=dataset_spec.dataset_type,
                name=dataset_spec.label,
                **dataset_spec.fixed_params,
            )

    # Convert accumulated results to DataFrames
    experiment.results.covert_to_df()
    return experiment.results
