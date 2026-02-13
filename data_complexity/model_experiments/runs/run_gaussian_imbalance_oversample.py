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
from data_complexity.model_experiments.ml import (
    LogisticRegressionModel,
    SVMModel,
    RandomForestModel,
    KNNModel,
)


def random_oversample_balanced(X, y):
    """
    Randomly duplicate minority class samples until class distribution is 1:1.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    tuple
        New X and y arrays with balanced class distribution.
    """
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    X_parts = [X]
    y_parts = [y]

    for cls, count in zip(classes, counts):
        if count < majority_count:
            deficit = majority_count - count
            minority_idx = np.where(y == cls)[0]
            oversample_idx = np.random.choice(minority_idx, size=deficit, replace=True)
            X_parts.append(X[oversample_idx])
            y_parts.append(y[oversample_idx])
    return np.concatenate(X_parts), np.concatenate(y_parts)


def systematic_oversample(X, y):
    """
    Duplicate each minority class sample exactly once (2x original minority count).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    tuple
        New X and y arrays with each minority sample duplicated once.
    """
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    X_parts = [X]
    y_parts = [y]

    for cls, count in zip(classes, counts):
        if count < majority_count:
            minority_idx = np.where(y == cls)[0]
            X_parts.append(X[minority_idx])
            y_parts.append(y[minority_idx])

    return np.concatenate(X_parts), np.concatenate(y_parts)


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
        fixed_params={"class_separation": 1.0, 
                      "cov_type": "spherical", 
                      "cov_scale": 1.0,
                      "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
                      },
        num_samples=400,
        train_size=0.5,
    ),
    vary_parameter=ParameterSpec(
        name="minority_reduce_scaler",
        values=[1, 2, 4, 8, 16],
        label_format="imbalance={value}x (Oversampled)",
    ),
    models=models,
    ml_metrics=["accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    cv_folds=5,
    plots=[PlotType.CORRELATIONS, PlotType.SUMMARY, PlotType.HEATMAP],
    correlation_target="best_accuracy",
    name="gaussian_imbalance_oversample",
    train_post_process=random_oversample_balanced,
    # test_post_process=systematic_oversample,
    
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run()
    exp.compute_correlations()
    exp.print_summary(top_n=10)
    exp.save()
