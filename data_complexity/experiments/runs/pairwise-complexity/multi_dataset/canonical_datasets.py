"""
Pairwise complexity experiment: Canonical synthetic dataset types.

One DatasetSpec per synthetic dataset type at a medium-complexity setting.
Compares how different structural geometries manifest in complexity metrics.

Dataset types:
  - Gaussian (well-separated spherical clusters)
  - Moons (low noise crescent shapes)
  - Circles (low noise concentric circles)
  - Blobs (isotropic Gaussian clusters)
  - XOR (linearly inseparable quadrant structure)

Results saved to results/pairwise_canonical_datasets/.
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
        DatasetSpec("Gaussian", {**COMMON, "class_separation": 3.0, "cov_type": "spherical", "cov_scale": 1.0}, label="Gaussian"),
        DatasetSpec("Moons", {**COMMON, "moons_noise": 0.1}, label="Moons"),
        DatasetSpec("Circles", {**COMMON, "circles_noise": 0.05}, label="Circles"),
        DatasetSpec("Blobs", {**COMMON}, label="Blobs"),
        DatasetSpec("XOR", {**COMMON}, label="XOR"),
    ],
    x_label="Dataset",
    cv_folds=5,
    name="pairwise_canonical_datasets",
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
