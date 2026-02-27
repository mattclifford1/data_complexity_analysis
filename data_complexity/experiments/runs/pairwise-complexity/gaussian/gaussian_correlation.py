"""
Pairwise complexity experiment: Gaussian feature correlation sweep.

Varies cov_correlation from -0.8 to 0.8 with correlated covariance type,
showing how complexity metrics co-vary as feature correlation changes.

Results saved to results/pairwise_gaussian_correlation/.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    RunMode,
    PlotType,
    PearsonCorrelation,
    SpearmanCorrelation,
    KendallTau,
    MutualInformation,
    EuclideanDistance,
    datasets_from_sweep,
)

config = ExperimentConfig(
    datasets=datasets_from_sweep(
        DatasetSpec(
            "Gaussian",
            {
                "num_samples": 400,
                "train_size": 0.5,
                "class_separation": 3.0,
                "cov_type": "symmetric",
                "equal_test": True,
            },
        ),
        ParameterSpec(
            "cov_correlation",
            [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8],
            "corr={value}",
        ),
    ),
    x_label="cov_correlation",
    cv_folds=5,
    name="pairwise_gaussian_correlation",
    run_mode=RunMode.COMPLEXITY_ONLY,
    plots=[
        PlotType.LINE_PLOT_COMPLEXITY_COMBINED,
        PlotType.DATASETS_OVERVIEW,
        PlotType.COMPLEXITY_DISTANCES,
    ],
    pairwise_distance_measures=[
        PearsonCorrelation(),
        SpearmanCorrelation(),
        KendallTau(),
        MutualInformation(),
        EuclideanDistance(),
    ],
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.compute_complexity_pairwise_distances()
    exp.save()
