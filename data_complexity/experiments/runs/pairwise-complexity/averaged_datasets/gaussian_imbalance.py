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

NAME = "gaussian-imbalance"
SAVE_DIR = Path(__file__).parent / "results" / NAME

base_config = ExperimentConfig(
    datasets=[],
    x_label="imbalance",
    cv_folds=5,
    run_mode=RunMode.COMPLEXITY_ONLY,
    plots=[
        PlotType.LINE_PLOT_COMPLEXITY_COMBINED,
        PlotType.COMPLEXITY_DISTANCES,
        PlotType.DATASETS_OVERVIEW,
    ],
    pairwise_distance_measures=[
        PearsonCorrelation(),
        SpearmanCorrelation(),
        CosineSimilarity(),

    ],
    # ml_metrics=["accuracy"],
)

dataset_groups = {}
for sep in [5.0, 2.0, 1.0, 0.5, 0.1]:
    dataset_groups[f"gaussian_sep_{sep}"] = datasets_from_sweep(
        DatasetSpec(
            "Gaussian",
            {"num_samples": 400, "train_size": 0.5, "class_separation": sep},
        ),
        ParameterSpec("minority_reduce_scaler", [1, 2, 4, 8, 16], "imbalance={value}x"),
    )

grouped_config = GroupedExperimentConfig(
    dataset_groups=dataset_groups,
    base_config=base_config,
    name=NAME,
    save_dir=SAVE_DIR,
)


if __name__ == "__main__":
    grouped = GroupedExperiment(grouped_config)
    grouped.run(n_jobs=-1)
    grouped.compute_averaged_pairwise_distances()
    grouped.save()
