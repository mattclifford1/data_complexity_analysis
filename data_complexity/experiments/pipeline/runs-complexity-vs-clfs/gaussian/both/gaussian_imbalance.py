"""
Example: Run Gaussian imbalance experiment with custom configuration.

Demonstrates how to study the effect of class imbalance on complexity metrics
and ML model performance. Uses minority_reduce_scaler to vary imbalance from
balanced (1x) to extreme imbalance (16x).

Imbalance is applied to the training set after the train/test split, so
complexity metrics reflect the actual imbalanced training data.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
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

# Configure experiment
config = ExperimentConfig(
    dataset=DatasetSpec(
        dataset_type="Gaussian",
        fixed_params={
            "num_samples": 400,
            "train_size": 0.5,
            "class_separation": 1.0, 
            "cov_type": 
            "spherical", 
            "cov_scale": 1.0,
            "equal_test": True, # Ensure test set is balanced for fair evaluation of imbalance effects
            },
    ),
    vary_parameter=ParameterSpec(
        name="minority_reduce_scaler",
        values=[1, 2, 4, 8, 16],
        label_format="imbalance={value}x",
    ),
    models=models,
    ml_metrics=["accuracy", "f1", "precision", "recall", "balanced_accuracy"],
    cv_folds=5,
    plots=[
        PlotType.LINE_PLOT_MODELS_COMBINED,
        PlotType.LINE_PLOT_COMPLEXITY_COMBINED,
        PlotType.DATASETS_OVERVIEW,
        PlotType.COMPLEXITY_CORRELATIONS,
        PlotType.ML_CORRELATIONS,
        ],
    name="gaussian_imbalance_example",
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    # exp.compute_correlations()
    exp.compute_complexity_correlations(source='train')
    exp.compute_ml_correlations(source='train')
    # exp.print_summary(top_n=10)
    exp.save()
