"""
Example: Run Gaussian variance experiment with custom configuration.

Demonstrates how to configure ML models, metrics, and plot types.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    RunMode,
)

# Configure experiment
config = ExperimentConfig(
    dataset=DatasetSpec(
        dataset_type="Moons",
        fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            "moons_noise": 0.1,
            "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
            },
    ),
    vary_parameter=ParameterSpec(
        name="minority_reduce_scaler",
        values=[1, 2, 4, 8, 16],
        label_format="imbalance={value}x",
    ),
    name="moons_imbalance_complexity",
    run_mode=RunMode.COMPLEXITY_ONLY,
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.save()
