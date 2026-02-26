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
├── metrics.py                # Main wrapper class (complexity_metrics)
├── abstract_metrics.py       # Base class for custom metrics
├── plotting/                 # Visualization utilities
│   └── plot_multiple_datasets.py
└── experiments/
    ├── __init__.py           # Exports ComplexityCollection, DatasetEntry
    ├── pipeline/             # Experiment framework
    │   ├── __init__.py       # Public API
    │   ├── experiment.py     # Experiment class
    │   ├── runner.py         # Experiment loop & parallel worker
    │   ├── plotting.py       # Plot generation
    │   ├── io.py             # Save/load logic
    │   ├── legacy_complexity_over_datasets.py  # ComplexityCollection
    │   ├── config.py         # Pre-defined configs
    │   ├── utils.py          # Data classes & results container
    │   └── README.md         # Detailed framework docs
    ├── classification/       # ML evaluation module
    │   ├── models.py
    │   ├── classification_metrics.py
    │   ├── evaluation.py
    │   ├── model_pipeline.py
    │   └── __init__.py
    └── plotting.py           # Reusable plotting functions
```

Note: `model_experiments/` still exists with legacy experiment scripts.

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

# Classical measures (e.g. imbalance ratio)
classical = complexity.classical_measures_scalar()  # {'IR': imbalance_ratio}
classical_full = complexity.classical_measures_full()
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
- **Classical Measures (1):** IR (Imbalance Ratio)

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
- **`test_ml_evaluation.py`** - Tests for evaluation metrics and evaluators (35 tests)
- **`test_ml_pipeline.py`** - Tests for pipeline orchestration (25 tests)

## ML Model Evaluation

The `experiments/classification/` module provides a modular architecture for ML model evaluation with three main components: models, evaluation strategies, and orchestration.

### Quick Start

```python
from data_complexity.experiments.classification import evaluate_models, evaluate_single_model

# Evaluate all default models
results = evaluate_models({"X": X, "y": y})

# Evaluate a single model
from data_complexity.experiments.classification import LogisticRegressionModel
model = LogisticRegressionModel()
metrics = evaluate_single_model(model, {"X": X, "y": y}, cv_folds=5)
print(metrics["accuracy"]["mean"])
```

### Architecture

```
experiments/classification/
├── models.py           # Model classes (AbstractMLModel + 8 concrete models)
├── classification_metrics.py       # Metrics for classification (accuracy, F1, precision, recall, balanced accuracy etc.)
├── evaluation.py       # Evaluator classes e.g. cross validation
├── model_pipeline.py   # Orchestration functions
└── __init__.py         # Public API
```

**Models** are pure sklearn wrappers without evaluation logic.
**Classification Metrics** define how to score predictions (accuracy, F1, precision, etc.).
**Evaluators** define how to train and assess models (cross-validation, train-test split).
**Pipeline** provides high-level convenience functions.

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

### Model Factory Functions

```python
from data_complexity.experiments.classification import get_model_by_name, get_default_models

# Get model by name
model = get_model_by_name("svm", kernel="linear")

# Get all default models (10 instances)
models = get_default_models()
```

### Evaluation Metrics

```python
from data_complexity.experiments.classification import (
    AccuracyMetric,
    AccuracyBalancedMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
    get_default_metrics,
    get_metrics_dict,
)

# Use default metrics (all 5)
metrics = get_default_metrics()

# Use custom metrics
metrics = [AccuracyMetric(), F1Metric()]

# Get sklearn scoring dict
scoring = get_metrics_dict(metrics)  # {'accuracy': <callable>, 'f1': <callable>}
```

#### How get_metrics_dict Works

The `get_metrics_dict` function converts metric instances to sklearn-compatible scorers:
- All metrics are wrapped with `make_scorer` for simplicity and consistency.

```python
# Mix built-in and custom metrics
from data_complexity.experiments.classification import AccuracyMetric, AccuracyMinorityMetric, GeometricMeanMetric

metrics = [AccuracyMetric(), AccuracyMinorityMetric(), GeometricMeanMetric()]
scoring = get_metrics_dict(metrics)
# {'accuracy': <callable>, 'minority_accuracy': <callable>, 'geometric_mean': <callable>}

from sklearn.model_selection import cross_validate
results = cross_validate(model, X, y, cv=5, scoring=scoring)
# All metrics work correctly
```

#### Metric Factory Functions

For convenience, you can create metrics from string names:

```python
from data_complexity.experiments.classification import get_metric_by_name, get_metrics_from_names

# Single metric
metric = get_metric_by_name('accuracy')

# Multiple metrics from names
metrics = get_metrics_from_names(['accuracy', 'f1', 'precision', 'recall'])

# Available metric names:
# 'accuracy', 'balanced_accuracy', 'minority_accuracy', 'majority_accuracy',
# 'geometric_mean', 'geometric_mean_weighted', 'f1', 'f1_weighted',
# 'precision', 'precision_class_0', 'precision_class_1', 'precision_weighted',
# 'fscore', 'recall', 'recall_weighted', 'auc', 'roc_auc'
```

This is particularly useful when configuring experiments that accept metric names as strings.

### Evaluators

```python
from data_complexity.experiments.classification import (
    CrossValidationEvaluator,
    TrainTestSplitEvaluator,
    get_default_evaluator,
)

# Use cross-validation (default)
evaluator = CrossValidationEvaluator(cv_folds=5)
results = evaluator.evaluate(model, X, y)

# Use train-test split
evaluator = TrainTestSplitEvaluator(test_size=0.2)
results = evaluator.evaluate(model, X, y)

# Get default evaluator
evaluator = get_default_evaluator(cv_folds=5)
```

### Pipeline Orchestration

```python
from data_complexity.experiments.classification import (
    evaluate_models,
    evaluate_models_train_test,
    evaluate_single_model,
    get_best_metric,
    get_mean_metric,
    get_model_metric,
    print_evaluation_results,
)

# Evaluate multiple models with defaults (cross-validation)
results = evaluate_models(data)

# Evaluate with custom components
results = evaluate_models(
    data,
    models=[LogisticRegressionModel(), KNNModel()],
    metrics=[AccuracyMetric(), F1Metric()],
    evaluator=CrossValidationEvaluator(cv_folds=3),
)

# Train/test split evaluation (used by experiment framework)
train_results, test_results = evaluate_models_train_test(
    train_data={"X": X_train, "y": y_train},
    test_data={"X": X_test, "y": y_test},
    models=[LogisticRegressionModel()],
    metrics=[AccuracyMetric()],
)

# Extract metrics
best_acc = get_best_metric(results, "accuracy")
mean_acc = get_mean_metric(results, "accuracy")
lr_acc = get_model_metric(results, "LogisticRegression", "accuracy")

# Print formatted results
print_evaluation_results(results, "accuracy")
```

## Experiments

Experiment scripts in `data_complexity/experiments/` study how dataset parameters affect complexity metrics.

```
experiments/
├── __init__.py             # Exports ComplexityCollection, DatasetEntry
├── pipeline/               # Generic experiment framework
│   ├── experiment.py       # Experiment class
│   ├── runner.py           # Experiment loop & parallel worker
│   ├── plotting.py         # Plot generation
│   ├── io.py               # Save/load logic
│   ├── legacy_complexity_over_datasets.py  # ComplexityCollection
│   ├── config.py           # Pre-defined configs
│   ├── utils.py            # Data classes & results container
│   └── README.md           # Full framework docs
├── classification/         # ML evaluation module
└── plotting.py             # Reusable plotting functions
```

### Experiment Framework

The generic experiment framework provides a configurable, reusable approach to running complexity vs ML performance experiments. All experiments use **train/test splits**: for each dataset spec, data is split into train/test sets, complexity is computed on both, and ML models are trained on train and evaluated on both. Multiple random seeds (`cv_folds`) control reproducibility.

#### Quick Start

```python
from data_complexity.experiments.pipeline import Experiment, get_config, run_experiment

# Run a pre-defined experiment by name
exp = run_experiment("moons_noise")   # runs, computes correlations, saves

# Or get a config and run manually
config = get_config("gaussian_variance")
exp = Experiment(config)
exp.run(verbose=True, n_jobs=-1)   # parallel across dataset specs
exp.compute_correlations()
exp.print_summary()
exp.save()
```

#### Pre-defined Configurations

```python
from data_complexity.experiments.pipeline import (
    gaussian_variance_config,    # Vary Gaussian covariance scale
    gaussian_separation_config,  # Vary class separation distance
    gaussian_correlation_config, # Vary feature correlation
    gaussian_imbalance_config,   # Vary class imbalance ratio
    moons_noise_config,          # Vary moons noise level
    circles_noise_config,        # Vary circles noise level
    blobs_features_config,       # Vary dimensionality
    get_config,                  # Get config by name
    list_configs,                # List all available configs
    run_experiment,              # Run a named experiment
    run_all_experiments,         # Run all experiments
)

# List available configs
print(list_configs())
# ['gaussian_variance', 'gaussian_separation', 'gaussian_correlation',
#  'gaussian_imbalance', 'moons_noise', 'circles_noise', 'blobs_features']
```

#### Custom Experiments

`ExperimentConfig` takes a **list of `DatasetSpec`** objects. Each spec is one point on the x-axis of the resulting plots. Use `datasets_from_sweep` to generate specs from a parameter sweep:

```python
from data_complexity.experiments.pipeline import (
    Experiment,
    ExperimentConfig,
    DatasetSpec,
    ParameterSpec,
    PlotType,
    RunMode,
    datasets_from_sweep,
)

config = ExperimentConfig(
    datasets=datasets_from_sweep(
        DatasetSpec("Gaussian", {"cov_type": "spherical"}),
        ParameterSpec("class_separation", [1.0, 2.0, 3.0, 4.0, 5.0], label_format="sep={value}"),
    ),
    x_label="class_separation",
    cv_folds=5,           # Number of random seeds for train/test splitting
    ml_metrics=["accuracy", "f1"],
    correlation_target="best_accuracy",
    plots=[PlotType.CORRELATIONS, PlotType.SUMMARY, PlotType.LINE_PLOT_TRAIN, PlotType.LINE_PLOT_TEST,
           PlotType.LINE_PLOT_MODELS_TRAIN, PlotType.LINE_PLOT_MODELS_TEST],
    name="my_custom_experiment",
)

exp = Experiment(config)
exp.run(verbose=True)
correlations = exp.compute_correlations()
exp.save()  # Saves to experiments/results/my_custom_experiment/
```

#### Run Modes

```python
from data_complexity.experiments.pipeline import RunMode

# Only compute complexity (skip ML evaluation — much faster)
config = ExperimentConfig(datasets=..., run_mode=RunMode.COMPLEXITY_ONLY)

# Only evaluate ML models (skip complexity computation)
config = ExperimentConfig(datasets=..., run_mode=RunMode.ML_ONLY)

# Both (default)
config = ExperimentConfig(datasets=..., run_mode=RunMode.BOTH)
```

#### Parallel Execution

```python
exp.run(verbose=True, n_jobs=-1)   # use all CPU cores
exp.run(verbose=True, n_jobs=4)    # explicit worker count
```

#### Experiment Results

Results are organized into subfolders within `results/{experiment_name}/`:
- `experiment_metadata.json` - Complete experiment configuration and parameters
- `data/` - CSV files (see below)
- `plots-{name}/` - Analysis visualizations
- `datasets/` - Dataset visualization PNGs (one per dataset spec)

CSV files in `data/`:
- `train_complexity_metrics.csv` - Complexity metrics on training data
- `test_complexity_metrics.csv` - Complexity metrics on test data
- `train_ml_performance.csv` - ML performance on training data
- `test_ml_performance.csv` - ML performance on test data
- `complexity_metrics.csv` - Train complexity (backward compat alias)
- `ml_performance.csv` - Test ML performance (backward compat alias)
- `correlations.csv` - Complexity–ML correlation results
- `complexity_correlations.csv` - Pairwise complexity metric correlations
- `ml_correlations.csv` - Pairwise ML metric correlations
- `per_classifier_correlations.csv` - Per-classifier aggregated correlations

Results are stored in pandas DataFrames:

```python
# After running
exp.results.train_complexity_df # Complexity on training data
exp.results.test_complexity_df  # Complexity on test data
exp.results.train_ml_df         # ML performance on training data
exp.results.test_ml_df          # ML performance on test data
exp.results.complexity_df       # Train complexity (backward compat)
exp.results.ml_df               # Test ML performance (backward compat)
exp.results.correlations_df     # Correlation results

# Complexity–ML correlations (Pearson)
exp.compute_correlations(
    complexity_source="train",  # 'train' or 'test'
    ml_source="test",           # 'train' or 'test'
)

# Additional correlation methods
exp.compute_complexity_correlations(source="train")  # pairwise among complexity metrics
exp.compute_ml_correlations(source="test")           # pairwise among ML metrics
exp.compute_per_classifier_correlations()            # per-classifier aggregated correlations

# Load previous results (supports both new hierarchical and legacy flat structure)
exp.load_results(Path("results/gaussian_variance/"))
```

**Note:** `complexity_df` returns train complexity and `ml_df` returns test ML, matching the standard approach of correlating training data complexity with generalization performance.

#### Available Plot Types

```python
from data_complexity.experiments.pipeline import PlotType
```

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
| `CORRELATIONS` | Bar chart of top complexity–ML correlations |
| `COMPLEXITY_CORRELATIONS` | Heatmap of pairwise complexity metric correlations |
| `ML_CORRELATIONS` | Heatmap of pairwise ML metric correlations |
| `SUMMARY` | Combined summary panel |
| `HEATMAP` | Per-model correlation heatmap |

### ComplexityCollection

`ComplexityCollection` computes complexity metrics across multiple datasets without ML evaluation — useful for comparing real and synthetic datasets.

```python
from data_complexity.experiments.pipeline import ComplexityCollection

collection = ComplexityCollection(seeds=5, train_size=0.5)

# Add a real dataset
collection.add_dataset("iris", {"X": X, "y": y})

# Add a synthetic dataset
collection.add_synthetic("gaussian", "Gaussian", {"class_separation": 4.0})

# Add a parameter sweep
collection.add_synthetic_sweep("moons", "Moons", {}, "moons_noise", [0.05, 0.1, 0.2])

# Compute complexity metrics (returns DataFrame)
metrics_df = collection.compute()

# Correlation matrix among complexity metrics
corr_matrix = collection.compute_correlations()

# Heatmap figure
fig = collection.plot_heatmap()

# Save results
collection.save(Path("results/"))
```

### Legacy Experiments

The original experiment scripts in `model_experiments/` still work for backwards compatibility:

```bash
pdm run python data_complexity/model_experiments/exp_complexity_vs_ml.py
```

These train 10 classifiers via cross-validation and compute Pearson correlations between each complexity metric and ML accuracy. Outputs are saved to `model_experiments/results/`.

## coding style
always write clear and concise code.

## documentation
always document features in the relevant README.md files and in code docstrings. Use type hints for clarity.
