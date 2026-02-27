"""
Pairwise complexity experiment: Classic sklearn datasets.

Compares complexity metrics across three well-known sklearn toy datasets:
Iris, Wine, and Breast Cancer. Each is loaded and split 50/50 train/test.

Results saved to results/pairwise_sklearn_classic/.
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
        DatasetSpec("Iris", {"train_size": 0.5}, label="Iris"),
        DatasetSpec("Wine", {"train_size": 0.5}, label="Wine"),
        DatasetSpec("Breast Cancer", {"train_size": 0.5}, label="Breast Cancer"),
    ],
    x_label="Dataset",
    cv_folds=5,
    name="pairwise_sklearn_classic",
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
