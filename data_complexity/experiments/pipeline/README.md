# Experiment Pipeline

Generic framework for studying how data complexity metrics correlate with ML classifier performance. For each dataset in the experiment, multiple random seeds split the data into train/test; complexity is computed on both splits and ML models are trained on train and evaluated on both.

## Quick start — pre-defined experiments

```python
from data_complexity.experiments.pipeline import run_experiment, list_configs

print(list_configs())
# ['gaussian_variance', 'gaussian_separation', 'gaussian_correlation',
#  'gaussian_imbalance', 'moons_noise', 'circles_noise', 'blobs_features']

exp = run_experiment("moons_noise")           # runs, computes correlations, saves
exp = run_experiment("gaussian_variance", save=False)  # skip saving
```

Or get a config and run it manually for more control:

```python
from data_complexity.experiments.pipeline import get_config, Experiment

config = get_config("gaussian_variance")
exp = Experiment(config)
exp.run(verbose=True, n_jobs=-1)   # parallel across dataset specs
exp.compute_correlations()
exp.print_summary(top_n=10)
exp.save()
```

---

## Core concepts

### `DatasetSpec` — one dataset point

```python
from data_complexity.experiments.pipeline import DatasetSpec

# Explicit label
spec = DatasetSpec("Moons", {"moons_noise": 0.1}, label="noise=0.1")

# Auto-generated label: "<type>_<first_param>=<value>"
spec = DatasetSpec("Gaussian", {"cov_scale": 2.0})
# spec.label == "Gaussian_cov_scale=2.0"

# No params: label defaults to the dataset type
spec = DatasetSpec("Gaussian")
# spec.label == "Gaussian"
```

### `ExperimentConfig` — the experiment

`ExperimentConfig` takes a **list of `DatasetSpec`** objects. Each spec is one point on the x-axis of the resulting plots.

```python
from data_complexity.experiments.pipeline import ExperimentConfig, DatasetSpec

config = ExperimentConfig(
    datasets=[
        DatasetSpec("Moons", {"moons_noise": 0.05}, label="low noise"),
        DatasetSpec("Moons", {"moons_noise": 0.20}, label="mid noise"),
        DatasetSpec("Moons", {"moons_noise": 0.50}, label="high noise"),
    ],
    x_label="noise level",     # x-axis label on all plots
    name="moons_noise_demo",
    cv_folds=5,                # random seeds for train/test averaging
    ml_metrics=["accuracy", "f1"],
    distance_target="best_accuracy",
)
```

Key parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `datasets` | `List[DatasetSpec]` | required | Ordered dataset specs to iterate over |
| `x_label` | `str` | `"Dataset"` | X-axis label for plots |
| `cv_folds` | `int` | `5` | Random seeds for train/test splitting |
| `ml_metrics` | `List[str]` | `["accuracy", "f1"]` | ML metrics to compute |
| `models` | `List[AbstractMLModel]` | `None` | ML models (default: all 10) |
| `distance_target` | `str` | `"best_accuracy"` | Target ML metric for distance summary |
| `run_mode` | `RunMode` | `BOTH` | `BOTH`, `COMPLEXITY_ONLY`, or `ML_ONLY` |
| `plots` | `List[PlotType]` | all plots | Plot types to generate |
| `pairwise_distance_measures` | `List[DistanceBetweenMetrics]` | `[PearsonCorrelation()]` | Measures for pairwise heatmaps |
| `name` | `str` | auto | Experiment name (used for save directory) |
| `save_dir` | `Path` | `results/{name}/` | Where to save outputs |

---

## Parameter sweeps with `datasets_from_sweep`

For the common case of varying a single parameter over a fixed base dataset, use the `datasets_from_sweep` helper to build the list of specs:

```python
from data_complexity.experiments.pipeline import (
    datasets_from_sweep, DatasetSpec, ParameterSpec, ExperimentConfig,
)

config = ExperimentConfig(
    datasets=datasets_from_sweep(
        DatasetSpec("Gaussian", {"cov_type": "spherical", "class_separation": 4.0}),
        ParameterSpec("cov_scale", [0.5, 1.0, 2.0, 4.0], label_format="scale={value}"),
    ),
    x_label="cov_scale",
    name="gaussian_variance",
)
```

`datasets_from_sweep(base_spec, param_spec)` produces one `DatasetSpec` per value in `param_spec.values`. Each spec inherits `base_spec.fixed_params`, adds the swept parameter, and gets its label from `param_spec.format_label(value)`.

### `ParameterSpec`

```python
from data_complexity.experiments.pipeline import ParameterSpec

spec = ParameterSpec(
    name="cov_scale",                  # parameter name passed to get_dataset()
    values=[0.5, 1.0, 2.0, 4.0],      # values to sweep
    label_format="scale={value}",      # x-axis tick labels (use {name} and/or {value})
)
spec.format_label(2.0)  # → "scale=2.0"
```

---

## Mixing different dataset types

Because `datasets` is just a list, you can freely mix dataset types in one experiment:

```python
config = ExperimentConfig(
    datasets=[
        DatasetSpec("Moons",     {"moons_noise": 0.1},       label="moons-easy"),
        DatasetSpec("Circles",   {"circles_noise": 0.05},    label="circles-easy"),
        DatasetSpec("Gaussian",  {"class_separation": 4.0},  label="gaussian-sep4"),
        DatasetSpec("Gaussian",  {"class_separation": 1.5},  label="gaussian-sep1.5"),
    ],
    x_label="Dataset",
    name="mixed_datasets",
)
```

---

## Running an experiment

```python
exp = Experiment(config)

# Sequential (default)
results = exp.run(verbose=True)

# Parallel — one process per dataset spec
results = exp.run(verbose=True, n_jobs=-1)

# Explicit worker count
results = exp.run(verbose=True, n_jobs=4)
```

`run()` returns an `ExperimentResultsContainer` and also stores it on `exp.results`.

---

## Accessing results

```python
# Train/test DataFrames (one row per dataset spec)
exp.results.train_complexity_df   # complexity metrics on training data
exp.results.test_complexity_df    # complexity metrics on test data
exp.results.train_ml_df           # ML performance on training data
exp.results.test_ml_df            # ML performance on test data

# Backwards-compat aliases
exp.results.complexity_df         # → train_complexity_df
exp.results.ml_df                 # → test_ml_df

# The param_value column contains label strings (e.g. "scale=1.0")
print(exp.results.train_complexity_df["param_value"].tolist())
# → ['scale=0.5', 'scale=1.0', 'scale=2.0', 'scale=4.0']

# Loaded dataset objects (dict keyed by label string)
exp.datasets["scale=2.0"]    # the dataset loader for that spec
```

---

## Correlations

```python
# Pearson correlation between complexity metrics and ML performance
corr_df = exp.compute_correlations(
    complexity_source="train",  # 'train' or 'test'
    ml_source="test",           # 'train' or 'test'
    ml_column="best_accuracy",  # optional override for correlation_target
)

# Pairwise distances among complexity metrics (returns dict: name -> N×N DataFrame)
# Uses config.pairwise_distance_measures by default, or pass a custom list
pairwise = exp.compute_complexity_pairwise_distances()
# pairwise["pearson_r"]      -> N×N matrix
# pairwise["spearman_rho"]   -> N×N matrix

# Pairwise distances among ML metrics
ml_pairwise = exp.compute_ml_pairwise_distances()

# Per-classifier aggregated correlations
per_clf = exp.compute_per_classifier_distances()

# Human-readable summary
exp.print_summary(top_n=10)
```

### Multiple pairwise measures

Pass `pairwise_distance_measures` in `ExperimentConfig` to compute heatmaps with multiple association measures in one call:

```python
from data_complexity.experiments.pipeline import (
    ExperimentConfig, datasets_from_sweep,
    PearsonCorrelation, SpearmanCorrelation, KendallTau,
)

config = ExperimentConfig(
    datasets=datasets_from_sweep(...),
    pairwise_distance_measures=[
        PearsonCorrelation(),
        SpearmanCorrelation(),
        KendallTau(),
    ],
)

exp = Experiment(config)
exp.run()
exp.compute_complexity_pairwise_distances()  # computes all 3 measures at once
exp.save()  # writes one PNG + one CSV per measure per source
```

Available measures (importable from `data_complexity.experiments.pipeline`):

| Class | `name` | Range | Signed | Description |
|---|---|---|---|---|
| `PearsonCorrelation` | `pearson_r` | [-1, 1] | yes | Pearson product-moment r |
| `SpearmanCorrelation` | `spearman_rho` | [-1, 1] | yes | Spearman rank ρ |
| `KendallTau` | `kendall_tau` | [-1, 1] | yes | Kendall τ |
| `MutualInformation` | `mutual_information` | [0, ∞) | no | k-NN mutual information |
| `EuclideanDistance` | `euclidean_distance` | [0, ∞) | no | Euclidean (z-score normalised) |
| `DistanceCorrelation` | `distance_correlation` | [0, 1] | no | Distance correlation (dcor) |
| `CosineSimilarity` | `cosine_similarity` | [-1, 1] | yes | Cosine similarity |
| `ManhattanDistance` | `manhattan_distance` | [0, ∞) | no | Manhattan L1 (z-score normalised) |

---

## Distance measures reference

Full reference for all 8 available association measures.

| Class | `name` | Range | Signed | p-value |
|---|---|---|---|---|
| `PearsonCorrelation` | `pearson_r` | [-1, 1] | yes | yes |
| `SpearmanCorrelation` | `spearman_rho` | [-1, 1] | yes | yes |
| `KendallTau` | `kendall_tau` | [-1, 1] | yes | yes |
| `MutualInformation` | `mutual_information` | [0, ∞) | no | no |
| `EuclideanDistance` | `euclidean_distance` | [0, ∞) | no | no |
| `DistanceCorrelation` | `distance_correlation` | [0, 1] | no | no |
| `CosineSimilarity` | `cosine_similarity` | [-1, 1] | yes | no |
| `ManhattanDistance` | `manhattan_distance` | [0, ∞) | no | no |

### Pearson r

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \sum_i (y_i - \bar{y})^2}}$$

Measures the strength and direction of the **linear** relationship between two variables. Values near ±1 indicate a strong linear relationship; 0 indicates no linear relationship. Sensitive to outliers. Reports a p-value for the null hypothesis that ρ = 0.

**Range:** [-1, 1] &nbsp;|&nbsp; **Signed:** yes &nbsp;|&nbsp; **Use when:** you expect a linear relationship and the data are roughly normally distributed.

### Spearman ρ

$$\rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}, \quad d_i = \text{rank}(x_i) - \text{rank}(y_i)$$

Pearson correlation applied to the **ranks** of the data rather than the raw values. Captures monotonic (not just linear) relationships and is robust to outliers and non-normal distributions.

**Range:** [-1, 1] &nbsp;|&nbsp; **Signed:** yes &nbsp;|&nbsp; **Use when:** the relationship may be monotonic but non-linear, or when data contain outliers.

### Kendall τ

$$\tau = \frac{N_c - N_d}{\frac{1}{2} n(n-1)}$$

where $N_c$ is the number of concordant pairs and $N_d$ is the number of discordant pairs. Measures the proportion of concordant minus discordant pairs. More robust than Spearman for small samples or many ties.

**Range:** [-1, 1] &nbsp;|&nbsp; **Signed:** yes &nbsp;|&nbsp; **Use when:** you need a robust rank-based measure with well-behaved statistics for small samples.

### Mutual Information

$$I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)\, p(y)}$$

Measures how much knowing one variable reduces uncertainty about the other. Captures **any** statistical dependency (linear or non-linear). Estimated from continuous data via k-nearest-neighbour density estimation. Always non-negative; zero only if the variables are independent.

**Range:** [0, ∞) &nbsp;|&nbsp; **Signed:** no &nbsp;|&nbsp; **Use when:** you want to detect any form of dependency, including complex non-linear relationships, without assuming a particular functional form.

### Euclidean Distance

$$d_E(x, y) = \left\| \tilde{x} - \tilde{y} \right\|_2 = \sqrt{\sum_i (\tilde{x}_i - \tilde{y}_i)^2}$$

where $\tilde{x}$ and $\tilde{y}$ are z-score normalised ($\mu = 0$, $\sigma = 1$). Measures geometric proximity in normalised space. Small values mean the two metric trajectories have similar shape and scale; large values mean they diverge.

**Range:** [0, ∞) &nbsp;|&nbsp; **Signed:** no &nbsp;|&nbsp; **Use when:** you want a simple geometric similarity that treats large deviations at any single point as important (L2 penalises outliers more than L1).

### Distance Correlation

$$\text{dCor}(X, Y) = \sqrt{\frac{\mathcal{V}^2(X, Y)}{\sqrt{\mathcal{V}^2(X, X)\, \mathcal{V}^2(Y, Y)}}}$$

where $\mathcal{V}^2$ is the distance covariance computed from pairwise Euclidean distances in the sample. Unlike Pearson r, distance correlation equals **zero if and only if** $X$ and $Y$ are statistically independent. It detects both linear and non-linear associations without any distributional assumptions. Requires the `dcor` package.

**Range:** [0, 1] &nbsp;|&nbsp; **Signed:** no &nbsp;|&nbsp; **Use when:** you want a dependence measure that is theoretically guaranteed to be zero only under independence, and you need to capture non-linear structure.

### Cosine Similarity

$$\cos(x, y) = \frac{x \cdot y}{\|x\|_2\, \|y\|_2} = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2}\, \sqrt{\sum_i y_i^2}}$$

Measures the **angle** between two vectors in the original (unnormalised) space, ignoring magnitude. Two metric trajectories with the same relative shape but different absolute scales receive a high cosine similarity. Sensitive to the mean level of each series — for mean-centred comparisons, Pearson r is equivalent.

**Range:** [-1, 1] &nbsp;|&nbsp; **Signed:** yes &nbsp;|&nbsp; **Use when:** you care about the direction/shape of the trajectories rather than their absolute magnitude.

### Manhattan Distance

$$d_M(x, y) = \left\| \tilde{x} - \tilde{y} \right\|_1 = \sum_i |\tilde{x}_i - \tilde{y}_i|$$

where $\tilde{x}$ and $\tilde{y}$ are z-score normalised. Like Euclidean distance but uses the L1 norm, which is **less sensitive to large individual deviations** because differences are not squared. Two trajectories that differ by a small constant amount at every point will accumulate a large L1 distance.

**Range:** [0, ∞) &nbsp;|&nbsp; **Signed:** no &nbsp;|&nbsp; **Use when:** you want a geometric similarity measure that is more robust to individual outlier points than Euclidean distance.

---

## Plotting

```python
from data_complexity.experiments.pipeline import PlotType

# Generate all configured plots
figures = exp.plot()

# Generate specific plots
figures = exp.plot(plot_types=[
    PlotType.LINE_PLOT_TEST,
    PlotType.LINE_PLOT_MODELS_TEST,
])
```

Available plot types:

| `PlotType` | Description |
|---|---|
| `LINE_PLOT_TRAIN` | Complexity + best-model accuracy vs x-axis (train) |
| `LINE_PLOT_TEST` | Complexity + best-model accuracy vs x-axis (test) |
| `LINE_PLOT_MODELS_TRAIN` | Per-model accuracy vs x-axis (train) |
| `LINE_PLOT_MODELS_TEST` | Per-model accuracy vs x-axis (test) |
| `LINE_PLOT_COMPLEXITY_TRAIN` | Each complexity metric vs x-axis (train) |
| `LINE_PLOT_COMPLEXITY_TEST` | Each complexity metric vs x-axis (test) |
| `LINE_PLOT_MODELS_COMBINED` | Train vs test per-model accuracy, side by side |
| `LINE_PLOT_COMPLEXITY_COMBINED` | Train vs test complexity, side by side |
| `DATASETS_OVERVIEW` | Grid of scatter plots for each dataset spec |
| `DISTANCES` | Bar chart of top complexity–ML distances |
| `COMPLEXITY_DISTANCES` | One heatmap per pairwise measure per source (train/test), saved under `complexity-distances/` |
| `ML_DISTANCES` | One heatmap per pairwise measure, saved under `ml-distances/` |
| `SUMMARY` | Combined summary panel |
| `HEATMAP` | Per-model correlation heatmap |

---

## Saving and loading

```python
# Save everything: CSVs, plots, dataset PNGs, metadata JSON
exp.save()                          # saves to config.save_dir
exp.save(save_dir=Path("/my/dir"))  # custom directory

# Saved structure:
# results/{name}/
#   experiment_metadata.json        ← config snapshot (includes pairwise_distance_measures)
#   data/
#     train_complexity_metrics.csv
#     test_complexity_metrics.csv
#     train_ml_performance.csv
#     test_ml_performance.csv
#     complexity_metrics.csv        ← alias for train (backwards compat)
#     ml_performance.csv            ← alias for test (backwards compat)
#     distances.csv
#     complexity_pairwise_distances_{name}.csv   ← one per measure (train)
#     complexity_pairwise_distances_test_{name}.csv  ← one per measure (test)
#     ml_pairwise_distances_{name}.csv           ← one per measure
#   plots-{name}/
#     line_plot_train.png, ...
#     complexity-distances/
#       pearson_r_train.png         ← one PNG per measure per source
#       pearson_r_test.png
#       spearman_rho_train.png
#     ml-distances/
#       pearson_r.png
#   datasets/
#     dataset_scale_1.0.png, ...

# Load previously saved results
exp2 = Experiment(config)
results = exp2.load_results(Path("results/my_experiment/"))
```

---

## Custom models and metrics

```python
from data_complexity.experiments.pipeline import ExperimentConfig, datasets_from_sweep, DatasetSpec, ParameterSpec
from data_complexity.experiments.classification import (
    LogisticRegressionModel, SVMModel, RandomForestModel,
    AccuracyMetric, F1Metric, AccuracyBalancedMetric,
)

config = ExperimentConfig(
    datasets=datasets_from_sweep(
        DatasetSpec("Moons", {}),
        ParameterSpec("moons_noise", [0.05, 0.1, 0.2, 0.3, 0.5], "noise={value}"),
    ),
    x_label="moons_noise",
    models=[
        LogisticRegressionModel(),
        SVMModel(kernel="rbf"),
        RandomForestModel(n_estimators=100),
    ],
    ml_metrics=["accuracy", "f1", "balanced_accuracy"],
    cv_folds=10,
    distance_target="best_balanced_accuracy",
    name="moons_custom",
)
```

---

## Available dataset types

| Type | Key parameters |
|---|---|
| `"Gaussian"` | `class_separation`, `cov_type`, `cov_scale`, `cov_correlation`, `num_samples`, `train_size`, `minority_reduce_scaler` |
| `"Moons"` | `moons_noise`, `num_samples`, `train_size` |
| `"Circles"` | `circles_noise`, `num_samples`, `train_size` |
| `"Blobs"` | `blobs_features`, `num_samples`, `train_size` |
| `"XOR"` | `num_samples`, `train_size` |

---

## Run modes

```python
from data_complexity.experiments.pipeline import RunMode

# Only compute complexity (skip ML evaluation — much faster)
config = ExperimentConfig(datasets=..., run_mode=RunMode.COMPLEXITY_ONLY)

# Only evaluate ML models (skip complexity computation)
config = ExperimentConfig(datasets=..., run_mode=RunMode.ML_ONLY)

# Both (default)
config = ExperimentConfig(datasets=..., run_mode=RunMode.BOTH)
```
