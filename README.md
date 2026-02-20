# data_complexity_analysis

Python library for analyzing data complexity metrics in classification datasets. Wraps [PyCol](https://github.com/brunoback/pycol) (Python Class Overlap Library) and adds an experiment framework for studying how complexity metrics correlate with ML classifier performance.

## What it does

- Computes 33+ complexity metrics across four categories: Feature Overlap, Instance Overlap, Structural Overlap, and Multiresolution Overlap
- Provides a modular ML evaluation module (8 classifiers, 17 metrics, cross-validation and train/test evaluators)
- Includes a configurable experiment framework for parameter sweeps over synthetic datasets (Gaussian, Moons, Circles, Blobs)
- Supports parallel execution, result saving/loading, and a range of visualizations

## Installation

```bash
pdm install
```

## Quick start

```python
from data_complexity.metrics import complexity_metrics
import numpy as np

dataset = {"X": np.random.randn(200, 2), "y": np.array([0] * 100 + [1] * 100)}
complexity = complexity_metrics(dataset=dataset)

print(complexity.get_all_metrics_scalar())
```

Run a pre-defined experiment:

```python
from data_complexity.experiments.pipeline import run_experiment

exp = run_experiment("moons_noise")   # runs, saves plots and CSVs
```

## Further reading

- `CLAUDE.md` — full API reference for contributors and AI assistants
- `data_complexity/experiments/pipeline/README.md` — detailed experiment framework docs
