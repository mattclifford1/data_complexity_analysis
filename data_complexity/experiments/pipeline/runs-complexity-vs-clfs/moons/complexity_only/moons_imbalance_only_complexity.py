"""
Example: Run Gaussian variance experiment with custom configuration.

Demonstrates how to configure ML models, metrics, and plot types.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    RunMode,
)

fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            "moons_noise": 0.1,
            "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
            }
datasets = []
for imbalance_factor in [1, 2, 4, 8, 16]:
    dataset_params = fixed_params.copy()
    dataset_params["minority_reduce_scaler"] = imbalance_factor
    datasets.append(DatasetSpec("Moons", dataset_params, label=f"imbalance={imbalance_factor}x"))


# Configure experiment
config = ExperimentConfig(
    datasets=datasets,
    name="moons_imbalance_complexity",
    run_mode=RunMode.COMPLEXITY_ONLY,
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.save()
