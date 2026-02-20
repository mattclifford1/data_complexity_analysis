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
        dataset_type="Gaussian",
        fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            "class_separation": 4.0, 
            "cov_type": "spherical",
            "equal_test": True, # Ensure test set is balanced for fair evaluation of variance effects
            },
    ),
    vary_parameter=ParameterSpec(
        name="cov1_scaler",
        values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        label_format="scale={value}",
    ),
    name="gaussian_variance_1_class_complexity",
    run_mode=RunMode.COMPLEXITY_ONLY,
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.save()
