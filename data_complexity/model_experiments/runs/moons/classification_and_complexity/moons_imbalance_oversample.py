"""
Example: Run Gaussian variance experiment with custom configuration.

Demonstrates how to configure ML models, metrics, and plot types.
"""
from data_complexity.model_experiments.experiment import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    RunMode,
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

# Configure experiment
config = ExperimentConfig(
    dataset=DatasetSpec(
        dataset_type="Moons",
        fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            "moons_noise": 0.1,
            "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
            "train_post_process": RandomDuplicateMinorityUpsampler(factor="equal"),
            },
    ),
    vary_parameter=ParameterSpec(
        name="minority_reduce_scaler",
        values=[1, 2, 4, 8, 16],
        label_format="imbalance={value}x",
    ),
    run_mode=RunMode.BOTH,
    models=models,
    ml_metrics=["accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    name="moons_imbalance_oversample",
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.save()
