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

# Configure experiment
config = ExperimentConfig(
    dataset=DatasetSpec(
        dataset_type="Moons",
        fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            # "moons_noise": 0.1,
            "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
            },
    ),
    vary_parameter=ParameterSpec(
        name="moons_noise",
        values=[0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
        label_format="scale={value}",
    ),
    name="moons_noise_complexity",
    run_mode=RunMode.COMPLEXITY_ONLY,
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.save()
