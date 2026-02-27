"""
Pairwise complexity experiment: All datasets mega-run.

Combines synthetic datasets at medium complexity settings with real datasets
from three groups (sklearn classic, medical, real-world). Provides a broad
view of how complexity metrics vary across fundamentally different data types.

Synthetic datasets (5):
  - Gaussian, Moons, Circles, Blobs, XOR

Real datasets (9):
  - Sklearn classic: Iris, Wine, Breast Cancer
  - Medical: Diabetes, Heart Disease, Habermans
  - Real-world: Banknote, Ionosphere, Sonar

Results saved to results/pairwise_all_datasets/.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    RunMode,
    PlotType,
    get_all_measures,
)

SYNTH = {"num_samples": 400, "train_size": 0.5, "equal_test": True}

config = ExperimentConfig(
    datasets=[
        # Synthetic
        DatasetSpec("Gaussian", {**SYNTH, "class_separation": 3.0, "cov_type": "spherical", "cov_scale": 1.0}, label="Gaussian"),
        DatasetSpec("Moons", {**SYNTH, "moons_noise": 0.1}, label="Moons"),
        DatasetSpec("Circles", {**SYNTH, "circles_noise": 0.05}, label="Circles"),
        DatasetSpec("Blobs", {**SYNTH}, label="Blobs"),
        DatasetSpec("XOR", {**SYNTH}, label="XOR"),
        # Sklearn classic
        DatasetSpec("Iris", {"train_size": 0.5}, label="Iris"),
        DatasetSpec("Wine", {"train_size": 0.5}, label="Wine"),
        DatasetSpec("Breast Cancer", {"train_size": 0.5}, label="Breast Cancer"),
        # Medical
        DatasetSpec("Diabetes Pima Indian", {"train_size": 0.5}, label="Diabetes"),
        DatasetSpec("Heart Disease", {"train_size": 0.5}, label="Heart Disease"),
        DatasetSpec("Habermans Breast Cancer", {"train_size": 0.5}, label="Habermans"),
        # Real-world
        DatasetSpec("Banknote Authentication", {"train_size": 0.5}, label="Banknote"),
        DatasetSpec("Ionosphere", {"train_size": 0.5}, label="Ionosphere"),
        DatasetSpec("Sonar Rocks vs Mines", {"train_size": 0.5}, label="Sonar"),
    ],
    x_label="Dataset",
    cv_folds=5,
    name="pairwise_all_datasets",
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
