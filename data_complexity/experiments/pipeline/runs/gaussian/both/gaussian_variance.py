"""
Example: Run Gaussian variance experiment with custom configuration.

Demonstrates how to configure ML models, metrics, and plot types.
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
    "class_separation": 4.0,
    "cov_type": "spherical",
    "equal_test": True,  # Ensure test set is balanced for fair evaluation of variance effects
}
datasets = []
for value in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    dataset_params = fixed_params.copy()
    dataset_params["cov_scale"] = value
    datasets.append(DatasetSpec("Gaussian", dataset_params, label=f"scale={value}"))

# Configure experiment
config = ExperimentConfig(
    datasets=datasets,
    models=models,
    ml_metrics=["accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    cv_folds=5,
    correlation_target="best_accuracy",
    name="gaussian_variance_example",
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.compute_correlations()
    exp.print_summary(top_n=10)
    exp.save()
