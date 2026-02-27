"""
Example: Run Gaussian imbalance experiment with post-processing hooks.

Demonstrates how to use train_postprocess and test_postprocess to apply
data transformations after the train/test split and before complexity
computation and ML training/evaluation.

Two oversampling strategies are shown:
- random_oversample_balanced: applied to train set, duplicates minority
  class samples until 1:1 class balance is achieved.
- systematic_oversample: applied to test set, duplicates each minority
  class sample exactly once (2x the original minority count).
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    PlotType,
)
from data_complexity.experiments.classification import (
    LogisticRegressionModel,
    SVMModel,
    RandomForestModel,
    KNNModel,
)
from data_loaders.resampling import RandomDuplicateMinorityUpsampler


# Configure custom models (subset of available models)
models = [
    LogisticRegressionModel(),
    SVMModel(kernel="rbf"),
    SVMModel(kernel="linear"),
    RandomForestModel(n_estimators=50),
    KNNModel(n_neighbors=5),
]

fixed_params = {
    "num_samples": 400,
    "train_size": 0.5,
    "class_separation": 1.0,
    "cov_type": "spherical",
    "cov_scale": 1.0,
    "equal_test": True,  # Ensure test set is balanced for fair evaluation of imbalance effects
    "minority_reduce_scaler": 5,
    #   "test_post_process": systematic_oversample,
}
datasets = []
for factor in [1, 2, 4, 8, 16]:
    dataset_params = fixed_params.copy()
    dataset_params["train_post_process"] = RandomDuplicateMinorityUpsampler(factor=factor)
    datasets.append(DatasetSpec("Gaussian", dataset_params, label=f"Random Oversample Balanced (factor={factor})"))

# Configure experiment (mirrors run_gaussian_imbalance.py with postprocessors added)
config = ExperimentConfig(
    datasets=datasets,
    models=models,
    ml_metrics=[
        "accuracy",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
    ],
    cv_folds=5,
    # correlation_target="best_accuracy",
    name="gaussian_imbalance_vary_oversampling",
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    # exp.compute_correlations()
    exp.print_summary(top_n=10)
    exp.save()
