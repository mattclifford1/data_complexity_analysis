"""
Pairwise complexity experiment: Blobs dimensionality sweep.

Varies blobs_features from 2 to 64, showing how complexity metrics
co-vary as the number of features increases in isotropic Gaussian clusters.

Results saved to results/pairwise_blobs_features/.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    RunMode,
    PlotType,
    get_all_measures,
    datasets_from_sweep,
)

config = ExperimentConfig(
    datasets=datasets_from_sweep(
        DatasetSpec(
            "Blobs",
            {
                "num_samples": 400,
                "train_size": 0.5,
                "equal_test": True,
            },
        ),
        ParameterSpec("blobs_features", [2, 4, 8, 16, 32, 64], "d={value}"),
    ),
    x_label="blobs_features",
    cv_folds=5,
    name="pairwise_blobs_features",
    run_mode=RunMode.COMPLEXITY_ONLY,
    plots=[
        PlotType.LINE_PLOT_COMPLEXITY_COMBINED,
        PlotType.DATASETS_OVERVIEW,
        PlotType.COMPLEXITY_DISTANCES,
    ],
    pairwise_distance_measures=get_all_measures(),
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.compute_complexity_pairwise_distances()
    exp.save()
