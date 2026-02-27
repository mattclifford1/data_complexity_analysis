"""
Pairwise complexity experiment: Moons noise sweep.

Varies moons_noise from 0.05 to 2.0, showing how complexity metrics
co-vary as the moons dataset becomes increasingly noisy.

Results saved to results/pairwise_moons_noise/.
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
            "Moons",
            {
                "num_samples": 400,
                "train_size": 0.5,
                "equal_test": True,
            },
        ),
        ParameterSpec(
            "moons_noise",
            [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
            "noise={value}",
        ),
    ),
    x_label="moons_noise",
    cv_folds=5,
    name="pairwise_moons_noise",
    run_mode=RunMode.COMPLEXITY_ONLY,
    plots=[
        PlotType.LINE_PLOT_COMPLEXITY_COMBINED,
        PlotType.DATASETS_OVERVIEW,
        PlotType.COMPLEXITY_CORRELATIONS,
    ],
    pairwise_distance_measures=get_all_measures(),
)

if __name__ == "__main__":
    exp = Experiment(config)
    exp.run(n_jobs=-1)
    exp.compute_complexity_pairwise_distances()
    exp.save()
