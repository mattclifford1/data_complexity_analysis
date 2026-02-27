"""
Pairwise complexity experiment: Gaussian class imbalance sweep.

Varies minority_reduce_scaler from 1 (balanced) to 16 (extreme imbalance),
showing how complexity metrics co-vary as the training set becomes imbalanced.

Results saved to results/pairwise_gaussian_imbalance/.
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
                "class_separation": 2.0,
                "cov_scale": 1.0,
                "equal_test": True,
            },
        ),
        ParameterSpec("minority_reduce_scaler", [1, 2, 4, 8, 16], "imbalance={value}x"),
    ),
    x_label="minority_reduce_scaler",
    cv_folds=5,
    name="pairwise_gaussian_imbalance",
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
