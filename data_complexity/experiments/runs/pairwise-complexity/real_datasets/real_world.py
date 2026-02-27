"""
Pairwise complexity experiment: Real-world classification datasets.

Compares complexity metrics across five real-world classification datasets.
Each is loaded and split 50/50 train/test.

Datasets:
  - Banknote Authentication
  - Wheat Seeds
  - Ionosphere
  - Sonar Rocks vs Mines
  - Abalone Gender

Results saved to results/pairwise_real_world/.
"""
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    RunMode,
    PlotType,
    get_all_measures,
)

config = ExperimentConfig(
    datasets=[
        DatasetSpec("Banknote Authentication", {"train_size": 0.5}, label="Banknote"),
        DatasetSpec("Wheat Seeds", {"train_size": 0.5}, label="Wheat Seeds"),
        DatasetSpec("Ionosphere", {"train_size": 0.5}, label="Ionosphere"),
        DatasetSpec("Sonar Rocks vs Mines", {"train_size": 0.5}, label="Sonar"),
        DatasetSpec("Abalone Gender", {"train_size": 0.5}, label="Abalone"),
    ],
    x_label="Dataset",
    cv_folds=5,
    name="pairwise_real_world",
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
