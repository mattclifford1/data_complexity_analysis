"""
Pairwise complexity experiment: Gaussian covariance scale sweep.

Varies cov_scale from 0.5 to 5.0 with fixed class separation, showing
how complexity metrics co-vary as dataset variance increases.

Results saved to results/pairwise_gaussian_variance/.
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
                "class_separation": 4.0,
                "cov_type": "spherical",
                "equal_test": True,
            },
        ),
        ParameterSpec("cov_scale", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], "scale={value}"),
    ),
    x_label="cov_scale",
    cv_folds=5,
    name="pairwise_gaussian_variance",
    run_mode=RunMode.COMPLEXITY_ONLY,
    plots=[
        PlotType.LINE_PLOT_COMPLEXITY_COMBINED,
        PlotType.DATASETS_OVERVIEW,
        PlotType.COMPLEXITY_CORRELATIONS,
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
