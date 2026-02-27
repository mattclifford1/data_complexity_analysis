"""
Pairwise complexity experiment: Gaussian class separation sweep.

Varies class_separation from 1.0 to 8.0 with fixed variance, showing
how complexity metrics co-vary as classes become better separated.

Results saved to results/pairwise_gaussian_separation/.
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
            "Gaussian",
            {
                "num_samples": 400,
                "train_size": 0.5,
                "cov_scale": 1.0,
                "cov_type": "spherical",
                "equal_test": True,
            },
        ),
        ParameterSpec("class_separation", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], "sep={value}"),
    ),
    x_label="class_separation",
    cv_folds=5,
    name="pairwise_gaussian_separation",
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
