"""
Grouped pairwise complexity experiment: class separation averaged across dataset geometries.

Sweeps class difficulty (separation or noise) across four dataset groups and
averages the pairwise complexity distance matrices. Shows which complexity
metrics consistently co-vary as class separability changes, regardless of
how that separability is induced.

Results saved to results/grouped_separation_sweep/.
"""
from pathlib import Path

from data_complexity.experiments.pipeline import (
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    RunMode,
    PlotType,
    PearsonCorrelation,
    SpearmanCorrelation,
    CosineSimilarity,
    datasets_from_sweep,
)
from data_complexity.experiments.pipeline.grouped_experiment import (
    GroupedExperiment,
    GroupedExperimentConfig,
)

SAVE_DIR = Path(__file__).parent / "results" / "synthetic-imbalance"

base_config = ExperimentConfig(
    datasets=[],
    x_label="imbalance",
    cv_folds=5,
    run_mode=RunMode.COMPLEXITY_ONLY,
    plots=[
        PlotType.LINE_PLOT_COMPLEXITY_COMBINED,
        PlotType.COMPLEXITY_DISTANCES,
    ],
    pairwise_distance_measures=[
        PearsonCorrelation(),
        SpearmanCorrelation(),
        CosineSimilarity(),

    ],
    # ml_metrics=["accuracy"],
)

grouped_config = GroupedExperimentConfig(
    dataset_groups={
        "moons": datasets_from_sweep(
            DatasetSpec("Moons", {"num_samples": 400, "train_size": 0.5}),
            ParameterSpec("minority_reduce_scaler", [1, 2, 4, 8, 16], "imbalance={value}x"),
        ),
        "circles": datasets_from_sweep(
            DatasetSpec("Circles", {"num_samples": 400, "train_size": 0.5}),
            ParameterSpec("minority_reduce_scaler", [1, 2, 4, 8, 16], "imbalance={value}x"),
        ),
        "gaussian": datasets_from_sweep(
            DatasetSpec("Gaussian", {"num_samples": 400, "train_size": 0.5}),
            ParameterSpec("minority_reduce_scaler", [1, 2, 4, 8, 16], "imbalance={value}x"),
        ),
    },
    base_config=base_config,
    name="grouped_separation_sweep",
    save_dir=SAVE_DIR,
)


if __name__ == "__main__":
    grouped = GroupedExperiment(grouped_config)
    grouped.run(n_jobs=-1)
    grouped.compute_averaged_pairwise_distances()
    grouped.save()
