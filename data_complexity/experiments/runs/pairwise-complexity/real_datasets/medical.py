"""
Pairwise complexity experiment: Medical/health real datasets.

Compares complexity metrics across five medical and health classification
datasets. Each is loaded and split 50/50 train/test.

Datasets:
  - Diabetes Pima Indian
  - Heart Disease
  - Habermans Breast Cancer
  - Hepatitis
  - Chronic Kidney Disease

Results saved to results/pairwise_medical/.
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
        DatasetSpec("Diabetes Pima Indian", {"train_size": 0.5}, label="Diabetes"),
        DatasetSpec("Heart Disease", {"train_size": 0.5}, label="Heart Disease"),
        DatasetSpec("Habermans Breast Cancer", {"train_size": 0.5}, label="Habermans"),
        DatasetSpec("Hepatitis", {"train_size": 0.5}, label="Hepatitis"),
        DatasetSpec("Chronic Kidney Disease", {"train_size": 0.5}, label="CKD"),
    ],
    x_label="Dataset",
    cv_folds=5,
    name="pairwise_medical",
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
