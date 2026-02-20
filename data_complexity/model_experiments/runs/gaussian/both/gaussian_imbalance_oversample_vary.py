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
import numpy as np

from data_complexity.model_experiments.experiment import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    PlotType,
)
from data_complexity.model_experiments.classification import (
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

# Configure experiment (mirrors run_gaussian_imbalance.py with postprocessors added)
config = ExperimentConfig(
    dataset=DatasetSpec(
        dataset_type="Gaussian",
        fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            "class_separation": 1.0, 
            "cov_type": "spherical", 
            "cov_scale": 1.0,
            "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
            "minority_reduce_scaler": 5,
        #   "test_post_process": systematic_oversample,
        },
    ),
    vary_parameter=ParameterSpec(
        name="train_post_process",
        values=[
            RandomDuplicateMinorityUpsampler(factor=1), 
            RandomDuplicateMinorityUpsampler(factor=2), 
            RandomDuplicateMinorityUpsampler(factor=4), 
            RandomDuplicateMinorityUpsampler(factor=8), 
            RandomDuplicateMinorityUpsampler(factor=16)
            ],
        label_format="Random Oversample Balanced (factor={value})",
    ),
    models=models,
    ml_metrics=[
        "accuracy", 
        "f1", 
        "precision", 
        "recall", 
        "balanced_accuracy"
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
