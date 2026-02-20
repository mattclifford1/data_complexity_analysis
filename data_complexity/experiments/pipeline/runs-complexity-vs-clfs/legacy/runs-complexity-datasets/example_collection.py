"""
Example: ComplexityCollection across synthetic and real datasets.

Outputs saved to example_collection_results/:
  - complexity_metrics.csv         — (n_datasets × n_metrics)
  - complexity_correlations.csv    — pairwise Pearson correlation matrix
  - complexity_correlations_heatmap.png
  - datasets_overview.png          — grid of all dataset scatter plots
  - datasets/dataset_<name>.png    — per-dataset plot (3-panel for synthetic,
                                     scatter for real)
"""
from pathlib import Path

import numpy as np

from data_complexity.experiments.pipeline import ComplexityCollection

# Example real dataset (replace with actual data)
X_example = np.random.randn(300, 2)
y_example = (X_example[:, 0] + X_example[:, 1] > 0).astype(int)

collection = (
    ComplexityCollection(seeds=5, train_size=0.5, name="example sample datasets")
    .add_dataset("real_example", {"X": X_example, "y": y_example})
    .add_synthetic("easy_gaussian", "Gaussian", {"class_separation": 4.0})
    .add_synthetic_sweep(
        base_name="moons",
        dataset_type="Moons",
        fixed_params={"n_samples": 500},
        vary_param="noise",
        values=[0.05, 0.1, 0.2, 0.3],
    )
)

metrics_df = collection.compute()           # (n_datasets × n_metrics)
corr_matrix = collection.compute_correlations()  # (n_metrics × n_metrics)

this_file_path = Path(__file__).resolve().parent
collection.save(this_file_path / "example_collection_results")
# Saves CSVs, heatmap, datasets_overview.png, and datasets/ subfolder