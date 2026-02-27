"""
Pairwise complexity experiment: Difficulty spectrum across dataset types.

Arranges datasets from easy (low complexity) to hard (high complexity)
by mixing dataset types and parameter settings. Reveals which complexity
metrics track difficulty consistently across structural geometries.

Spectrum (easy → hard):
  1. Easy Gaussian     — high class separation
  2. Medium Gaussian   — moderate separation
  3. Hard Gaussian     — low separation + high covariance
  4. Easy Moons        — very low noise
  5. Hard Moons        — high noise (classes heavily overlapping)
  6. Circles           — concentric circles (inherently non-linear)
  7. XOR               — linearly inseparable quadrant structure

Results saved to results/pairwise_difficulty_spectrum/.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    RunMode,
    PlotType,
    get_all_measures,
)

COMMON = {"num_samples": 400, "train_size": 0.5, "equal_test": True}

config = ExperimentConfig(
    datasets=[
        DatasetSpec(
            "Gaussian",
            {**COMMON, "class_separation": 6.0, "cov_type": "spherical", "cov_scale": 0.5},
            label="Easy Gaussian",
        ),
        DatasetSpec(
            "Gaussian",
            {**COMMON, "class_separation": 3.0, "cov_type": "spherical", "cov_scale": 1.0},
            label="Medium Gaussian",
        ),
        DatasetSpec(
            "Gaussian",
            {**COMMON, "class_separation": 1.0, "cov_type": "spherical", "cov_scale": 2.0},
            label="Hard Gaussian",
        ),
        DatasetSpec(
            "Moons",
            {**COMMON, "moons_noise": 0.05},
            label="Easy Moons",
        ),
        DatasetSpec(
            "Moons",
            {**COMMON, "moons_noise": 0.5},
            label="Hard Moons",
        ),
        DatasetSpec(
            "Circles",
            {**COMMON, "circles_noise": 0.05},
            label="Circles",
        ),
        DatasetSpec(
            "XOR",
            {**COMMON},
            label="XOR",
        ),
    ],
    x_label="Dataset",
    cv_folds=5,
    name="pairwise_difficulty_spectrum",
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
