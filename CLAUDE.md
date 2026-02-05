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

## Experiments

Experiment scripts in `data_complexity/experiments/` study how dataset parameters affect complexity metrics.

```bash
pdm run python data_complexity/experiments/synthetic/gaussian/exp_separation.py
```

```
experiments/
├── ml_evaluation.py                    # ML model training utilities
├── exp_complexity_vs_ml.py             # Correlate complexity with ML accuracy (Gaussian variance)
├── exp_separation_vs_ml.py             # Correlate complexity with ML accuracy (class separation)
├── exp_moons_vs_ml.py                  # Correlate complexity with ML accuracy (moons noise)
├── exp_comprehensive_correlation.py    # Combined analysis across all dataset types
├── synthetic/
│   ├── exp_compare_generators.py       # Compare all synthetic types
│   ├── gaussian/                       # Gaussian parameter studies
│   ├── moons/                          # Moons noise and sample studies
│   ├── circles/                        # Circles noise studies
│   ├── blobs/                          # Blobs dimensionality studies
│   └── xor/                            # XOR sample studies
└── real/
    ├── exp_compare_datasets.py         # Compare UCI datasets
    ├── exp_scaling.py                  # Feature scaling effects
    └── exp_dim_reduction.py            # PCA reduction effects
```

### Complexity vs ML Performance

Experiments correlating complexity metrics with classifier accuracy:

```bash
pdm run python data_complexity/experiments/exp_complexity_vs_ml.py
```

These train LogReg, KNN, DecisionTree, SVM, RandomForest via cross-validation and compute Pearson correlations between each complexity metric and best accuracy. Outputs correlation bar charts and scatter plots.

## coding style
always wise clear and consise code.

## documentation
always document features in the readme and docstrings. Use type hints for clarity.