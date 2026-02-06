# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python library for analyzing data complexity metrics in classification datasets. Implements PyCol (Python Class Overlap Library) which measures complexity through four categories: Feature Overlap, Instance Overlap, Structural Overlap, and Multiresolution Overlap.

## Development Environment

Always use PDM (Python Dependency Manager) for environment and dependency management.

- `pdm install` - Install dependencies and create virtual environment
- `pdm install -G dev` - Install with dev dependencies
- `pdm add <package>` - Add a new dependency
- `pdm run <command>` - Run command in the PDM environment
- `pdm add -e path/to/package --dev` - Install a package in development mode

## Architecture

```
data_complexity/
├── metrics.py              # Main wrapper class (complexity_metrics)
├── abstract_metrics.py     # Base class for custom metrics
├── plot_multiple_datasets.py  # Visualization utilities
├── experiments/            # Parameter study scripts
└── pycol/
    ├── complexity.py       # Core implementation (~3000 lines, 50+ measures)
    ├── dataset/            # Test datasets (ARFF, CSV, pickle)
    └── use_cases/          # Example implementations
```

### Core Pattern

```python
from data_complexity.metrics import complexity_metrics

# Dataset as dict with NumPy arrays
dataset = {'X': X, 'y': y}
complexity = complexity_metrics(dataset=dataset)

# Get all metrics
all_metrics = complexity.get_all_metrics_scalar()

# Or by category
feature = complexity.feature_overlap_scalar()
instance = complexity.instance_overlap_scalar()
structural = complexity.structural_overlap_scalar()
multiresolution = complexity.multiresolution_overlap_full()
```

### Direct PyCol Usage

```python
from data_complexity.pycol import Complexity

# From arrays
comp = Complexity(dataset={'X': X, 'y': y}, file_type="array")

# From file
comp = Complexity("path/to/dataset.arff", file_type="arff")

# Individual measures
f1 = comp.F1()
overlap = comp.deg_overlap()
```

## Metric Categories

- **Feature Overlap (6):** F1, F1v, F2, F3, F4, IN
- **Instance Overlap (13):** R-value, Raug, degOver, N3, SI, N4, kDN, D3, CM, wCM, dwCM, Borderline Examples, IPoints
- **Structural Overlap (9):** N1, T1, Clust, ONB, LSCAvg, DBC, N2, NSG, ICSV
- **Multiresolution Overlap (5):** MRCA, C1, C2, Purity, Neighbourhood Separability

## Testing

```bash
pdm run pytest tests/ -v           # Run all tests
pdm run pytest tests/ -v -k "feature"  # Run tests matching "feature"
```

Tests are in `tests/` (not `pycol/` which is external code):

- **`conftest.py`** - Fixtures providing test datasets (linearly separable, moons, high overlap, multiclass, high-dimensional)
- **`test_metrics.py`** - Tests for `complexity_metrics` wrapper class
- **`test_abstract_metrics.py`** - Tests for `abstract_metric` base class
- **`test_ml_models.py`** - Tests for ML model classes (31 tests)

## ML Model Evaluation

The `experiments/ml_models.py` module provides a class-based interface for ML model evaluation.

### Abstract Base Class

```python
from data_complexity.experiments.ml_models import (
    AbstractMLModel,
    LogisticRegressionModel,
    SVMModel,
    evaluate_models,
)

# Use a single model
model = LogisticRegressionModel()
metrics = model.evaluate({"X": X, "y": y}, cv_folds=5)
print(metrics["accuracy"]["mean"])

# Or evaluate multiple models
results = evaluate_models({"X": X, "y": y})  # Uses all 10 default models
```

### Available Model Classes

| Class | Name | Key Parameters |
|-------|------|----------------|
| `LogisticRegressionModel` | LogisticRegression | `max_iter` |
| `KNNModel` | KNN-{n} | `n_neighbors` |
| `DecisionTreeModel` | DecisionTree | `max_depth` |
| `SVMModel` | SVM-{kernel} | `kernel`, `C` |
| `RandomForestModel` | RandomForest | `n_estimators`, `max_depth` |
| `GradientBoostingModel` | GradientBoosting | `n_estimators`, `max_depth` |
| `NaiveBayesModel` | NaiveBayes | - |
| `MLPModel` | MLP | `hidden_layer_sizes`, `max_iter` |

### Factory Functions

```python
from data_complexity.experiments.ml_models import get_model_by_name, get_default_models

# Get model by name
model = get_model_by_name("svm", kernel="linear")

# Get all default models
models = get_default_models()  # Returns list of 10 model instances
```

## Experiments

Experiment scripts in `data_complexity/experiments/` study how dataset parameters affect complexity metrics.

```
experiments/
├── experiment.py           # Generic experiment framework
├── experiment_configs.py   # Pre-defined experiment configurations
├── plotting.py             # Reusable plotting functions
├── ml_models.py            # Abstract ML model classes
├── ml_evaluation.py        # Functional wrapper (backwards compat)
├── exp_complexity_vs_ml.py # Legacy: Gaussian variance experiment
├── exp_separation_vs_ml.py # Legacy: Class separation experiment
├── exp_moons_vs_ml.py      # Legacy: Moons noise experiment
├── exp_comprehensive_correlation.py  # Combined analysis
├── results/                # Output directory for CSVs and plots
├── synthetic/              # Synthetic dataset studies
└── real/                   # Real dataset studies
```

### Experiment Framework

The generic experiment framework provides a configurable, reusable approach to running complexity vs ML performance experiments.

#### Quick Start

```python
from data_complexity.experiments.experiment import Experiment
from data_complexity.experiments.experiment_configs import gaussian_variance_config

# Run a pre-defined experiment
exp = Experiment(gaussian_variance_config())
exp.run()
exp.compute_correlations()
exp.print_summary()
exp.save()
```

#### Pre-defined Configurations

```python
from data_complexity.experiments.experiment_configs import (
    gaussian_variance_config,    # Vary Gaussian covariance scale
    gaussian_separation_config,  # Vary class separation distance
    gaussian_correlation_config, # Vary feature correlation
    moons_noise_config,          # Vary moons noise level
    circles_noise_config,        # Vary circles noise level
    blobs_features_config,       # Vary dimensionality
    get_config,                  # Get config by name
    list_configs,                # List all available configs
    run_experiment,              # Run a named experiment
    run_all_experiments,         # Run all experiments
)

# Run by name
from data_complexity.experiments.experiment_configs import run_experiment
exp = run_experiment("gaussian_variance")

# List available configs
print(list_configs())
# ['gaussian_variance', 'gaussian_separation', 'gaussian_correlation',
#  'moons_noise', 'circles_noise', 'blobs_features']
```

#### Custom Experiments

```python
from data_complexity.experiments.experiment import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    PlotType,
)

# Define custom experiment
config = ExperimentConfig(
    dataset=DatasetSpec(
        dataset_type="Gaussian",
        fixed_params={"cov_type": "spherical"},
        num_samples=500,
    ),
    vary_parameter=ParameterSpec(
        name="class_separation",
        values=[1.0, 2.0, 3.0, 4.0, 5.0],
        label_format="sep={value}",
    ),
    cv_folds=5,
    ml_metrics=["accuracy", "f1"],
    correlation_target="best_accuracy",
    plots=[PlotType.CORRELATIONS, PlotType.SUMMARY],
    name="my_custom_experiment",
)

exp = Experiment(config)
exp.run(verbose=True)
correlations = exp.compute_correlations()
exp.save()  # Saves to experiments/results/my_custom_experiment/
```

#### Experiment Results

Results are organized into subfolders within `experiments/results/{experiment_name}/`:
- `data/` - CSV files (complexity_metrics.csv, ml_performance.csv, correlations.csv)
- `plots/` - Analysis visualizations (correlations.png, summary.png, heatmap.png)
- `datasets/` - Dataset visualization PNGs (one per parameter value)

Results are stored in pandas DataFrames:

```python
# After running
exp.results.complexity_df  # Complexity metrics per parameter value
exp.results.ml_df          # ML performance per parameter value
exp.results.correlations_df  # Correlation results

# Load previous results (supports both new hierarchical and legacy flat structure)
exp.load_results(Path("experiments/results/gaussian_variance/"))
```

**Note:** The framework maintains backwards compatibility with results saved in the legacy flat structure, so existing experiment results remain accessible.

### Legacy Experiments

The original experiment scripts still work for backwards compatibility:

```bash
pdm run python data_complexity/experiments/exp_complexity_vs_ml.py
```

These train 10 classifiers via cross-validation and compute Pearson correlations between each complexity metric and ML accuracy. Outputs are saved to `experiments/results/`.

## coding style
always wise clear and consise code.

## documentation
always document features in the readme and docstrings. Use type hints for clarity.