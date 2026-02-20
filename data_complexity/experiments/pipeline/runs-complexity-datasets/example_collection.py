

from pathlib import Path
from data_complexity.experiments import ComplexityCollection

collection = (
     ComplexityCollection(seeds=5, train_size=0.5)
    #  .add_dataset("iris", {"X": X_iris, "y": y_iris})
     .add_synthetic("easy_gaussian", "Gaussian", {"class_separation": 4.0})
     .add_synthetic_sweep(
         base_name="moons",
         dataset_type="Moons",
         fixed_params={"n_samples": 500},
         vary_param="noise",
         values=[0.05, 0.1, 0.2, 0.3],
     )
 )
metrics_df = collection.compute()          # (n_datasets × n_metrics)
corr_matrix = collection.compute_correlations()  # (n_metrics × n_metrics)
fig = collection.plot_heatmap()
# get this file path and save the collection results to a directory called "results/my_study/"
this_file_path = Path(__file__).resolve().parent

collection.save(this_file_path / "results" / "my_study")