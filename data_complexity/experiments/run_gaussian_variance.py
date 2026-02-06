"""
Example: Run Gaussian variance experiment with custom configuration.

Demonstrates how to configure ML models, metrics, and plot types.
"""
from data_complexity.experiments.experiment import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    PlotType,
)
from data_complexity.experiments.ml_models import (
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

# Configure experiment
config = ExperimentConfig(
    dataset=DatasetSpec(
        dataset_type="Gaussian",
        fixed_params={"class_separation": 4.0, "cov_type": "spherical"},
        num_samples=400,
    ),
    vary_parameter=ParameterSpec(
        name="cov_scale",
        values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        label_format="scale={value}",
    ),
    models=models,
    ml_metrics=["accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    cv_folds=5,
    plots=[PlotType.CORRELATIONS, PlotType.SUMMARY, PlotType.HEATMAP],
    correlation_target="best_accuracy",
    name="gaussian_variance_example",
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run()
    exp.compute_correlations()
    exp.print_summary(top_n=10)
    exp.save()
