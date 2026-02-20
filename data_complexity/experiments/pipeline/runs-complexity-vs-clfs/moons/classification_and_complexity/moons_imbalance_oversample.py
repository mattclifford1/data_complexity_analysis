"""
Example: Run moons imbalance experiment with oversampling post-processing.

Demonstrates how to configure ML models, metrics, and plot types.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    RunMode,
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
    "moons_noise": 0.1,
    "equal_test": True,  # Ensure test set is balanced for fair evaluation of imbalance effects
    "train_post_process": RandomDuplicateMinorityUpsampler(factor="equal"),
}
datasets = []
for value in [1, 2, 4, 8, 16]:
    dataset_params = fixed_params.copy()
    dataset_params["minority_reduce_scaler"] = value
    datasets.append(DatasetSpec("Moons", dataset_params, label=f"imbalance={value}x"))

# Configure experiment
config = ExperimentConfig(
    datasets=datasets,
    run_mode=RunMode.BOTH,
    models=models,
    ml_metrics=["accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    name="moons_imbalance_oversample",
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.save()
