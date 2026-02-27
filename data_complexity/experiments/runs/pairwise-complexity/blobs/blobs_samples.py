"""
Pairwise complexity experiment: Blobs sample size sweep.

Varies num_samples from 50 to 1600, showing how complexity metrics
co-vary as dataset size grows for isotropic Gaussian clusters.

Results saved to results/pairwise_blobs_samples/.
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
                "blobs_features": 4,
                "train_size": 0.5,
                "equal_test": True,
            },
        ),
        ParameterSpec("num_samples", [50, 100, 200, 400, 800, 1600], "n={value}"),
    ),
    x_label="num_samples",
    cv_folds=5,
    name="pairwise_blobs_samples",
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
