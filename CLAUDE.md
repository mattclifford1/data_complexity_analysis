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

Experiment scripts in `data_complexity/experiments/` study how dataset parameters affect complexity metrics using the Gaussian generator from `toy_datasets`:

```bash
pdm run python data_complexity/experiments/exp_gaussian_separation.py
```

| Experiment | Parameter | Description |
|------------|-----------|-------------|
| `exp_gaussian_separation.py` | `class_separation` | Distance between class means |
| `exp_gaussian_cov_scale.py` | `cov_scale` | Variance magnitude (higher → more overlap) |
| `exp_gaussian_cov_types.py` | `cov_type` | Shape: spherical, diagonal, symmetric, random |
| `exp_gaussian_correlation.py` | `cov_correlation` | Feature correlation in symmetric covariance |
| `exp_gaussian_imbalance.py` | `num_samples` | Class ratio effects (1:1 to 9:1) |
| `exp_gaussian_asymmetric_cov.py` | `cov_scale` per class | Different variances per class |

Each script has a `run_experiment()` function returning metrics dict, usable programmatically.

## coding style
always wise clear and consise code.