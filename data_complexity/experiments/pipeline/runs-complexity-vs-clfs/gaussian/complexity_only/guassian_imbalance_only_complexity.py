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
    datasets_from_sweep,
)

fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            "class_separation": 1.0, 
            "cov_type": 
            "spherical", 
            "cov_scale": 1.0,
            "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
            }
datasets = []
for imbalance_factor in [1, 2, 4, 8, 16]:
    dataset_params = fixed_params.copy()
    dataset_params["minority_reduce_scaler"] = imbalance_factor
    datasets.append(DatasetSpec("Gaussian", dataset_params, label=f"imbalance={imbalance_factor}x"))

# Configure experiment
config = ExperimentConfig(
    datasets=datasets,
    # vary_parameter=ParameterSpec(
    #     name="minority_reduce_scaler",
    #     values=[1, 2, 4, 8, 16],
    #     label_format="imbalance={value}x",
    # ),
    name="gaussian_imbalance_complexity",
    run_mode=RunMode.COMPLEXITY_ONLY,
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=1)
    exp.save()
