"""
Pairwise complexity experiment: Circles noise sweep.

Varies circles_noise from 0.02 to 0.3, showing how complexity metrics
co-vary as the concentric circles dataset becomes increasingly noisy.

Results saved to results/pairwise_circles_noise/.
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
    DistanceCorrelation,
    CosineSimilarity,
    ManhattanDistance,
    datasets_from_sweep,
)

config = ExperimentConfig(
    datasets=datasets_from_sweep(
        DatasetSpec(
            "Circles",
            {
                "num_samples": 400,
                "train_size": 0.5,
                "equal_test": True,
            },
        ),
        ParameterSpec(
            "circles_noise",
            [0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
            "noise={value}",
        ),
    ),
    x_label="circles_noise",
    cv_folds=5,
    name="pairwise_circles_noise",
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
        DistanceCorrelation(),
        CosineSimilarity(),
        ManhattanDistance(),
    ],
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.compute_complexity_pairwise_distances()
    exp.save()
