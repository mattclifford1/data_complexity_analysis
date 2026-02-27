# Data Complexity Metrics

This module implements complexity measures for classification datasets, grouped into six categories. Each measure quantifies a different aspect of how difficult a dataset is to classify.

---

## Categories

| Category | Count | What it measures | Details |
|---|---|---|---|
| [Feature Overlap](#feature-overlap) | 6 | Discriminability of individual features | [feature/README.md](feature/README.md) |
| [Instance Overlap](#instance-overlap) | 9 | Hardness of individual training instances | [instance/README.md](instance/README.md) |
| [Structural Overlap](#structural-overlap) | 9 | Topology of the class boundary | [structural/README.md](structural/README.md) |
| [Multiresolution Overlap](#multiresolution-overlap) | 5 | Class purity at multiple resolutions | [multiresolution/README.md](multiresolution/README.md) |
| [Classical Measures](#classical-measures) | 1 | Dataset-level statistics | [classical/README.md](classical/README.md) |
| [Distributional Measures](#distributional-measures) | 5 | Statistical distribution overlap and boundary geometry | [distributional/README.md](distributional/README.md) |

---

## Feature Overlap

These measures assess how well individual features or linear combinations of features can separate classes. See [feature/README.md](feature/README.md) for full equations and references.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Max Fisher's Discriminant Ratio | F1 | Best single linear discriminant | [0, 1] | Hard (normalised: 1 = worst) |
| Directional-vector Max Fisher's | F1v | Best linear projection | [0, 1] | Hard |
| Volume of Overlapping Region | F2 | Feature-space overlap fraction | [0, 1] | Hard |
| Max Individual Feature Efficiency | F3 | Best single feature discriminability | [0, 1] | Hard |
| Collective Feature Efficiency | F4 | Sequential feature elimination | [0, 1] | Hard |
| Input Noise | IN | Fraction of instances in cross-class domains | [0, 1] | Hard |

---

## Instance Overlap

These measures assess difficulty at the instance level — how many training points are in overlapping regions or are surrounded by opposite-class neighbours. See [instance/README.md](instance/README.md) for full equations and references.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Augmented R Value | Raug | Weighted neighbourhood class mixing | [0, 1] | Hard |
| Degree of Overlap | deg_overlap | Direct overlap count | [0, 1] | Hard |
| Error Rate of 1-NN | N3 | Leave-one-out 1-NN error | [0, 1] | Hard |
| Separability Index | SI | Fraction of cleanly-separated samples | [0, 1] | Easy (high = separable) |
| Non-linearity of 1-NN | N4 | Error on interpolated test set | [0, 1] | Hard |
| K-Disagreeing Neighbours | kDN | Mean neighbourhood disagreement | [0, 1] | Hard |
| Disjunct Size | D3 | Minority instances in k-neighbourhood | [0, ∞) | Hard |
| Class Complexity Measure | CM | Fraction of kDN > 0.5 samples | [0, 1] | Hard |
| Borderline Examples | Borderline | Fraction of boundary-zone instances | [0, 1] | Hard |

---

## Structural Overlap

These measures capture the topology of the class boundary and the geometric structure of clusters. See [structural/README.md](structural/README.md) for full equations and references.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Fraction of Borderline Points | N1 | Density of inter-class MST edges | [0, 1] | Hard |
| Fraction of Covering Hyperspheres | T1 | Complexity of hypersphere cover | [0, 1] | Hard |
| Number of Clusters | Clust | Class fragmentation via local sets | [0, 1] | Hard |
| Intra/Inter NN Distance Ratio | N2 | Class compactness relative to separation | [0, 1] | Hard |
| Overlap of Neighbourhood Balls | ONB | Mean inter-class fraction in greedy cover | [0, 1] | Hard |
| Local Set Average Cardinality | LSCAvg | Mean size of same-class local sets | [0, 1] | Easy (low = large local sets) |
| Decision Boundary Complexity | DBC | Inter-class fraction of sphere-MST edges | [0, 1] | Hard |
| Number of Same-class Groups | NSG | Mean instances per covering sphere | [0, ∞) | Easy (large groups) |
| Intra-class Spatial Variability | ICSV | Std deviation of sphere density | [0, ∞) | Hard |

---

## Multiresolution Overlap

These measures analyse class purity across multiple spatial resolutions, from coarse to fine. See [multiresolution/README.md](multiresolution/README.md) for full equations and references.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Multiresolution Class Aggregate | MRCA | Weighted impurity across resolutions | [0, 1] | Hard |
| Class Entropy | C1 | Mean neighbourhood class entropy | [0, 1] | Hard |
| Multi-resolution Complexity | C2 | Distance-weighted neighbourhood error | [0, 1] | Hard |
| Purity | Purity | Hypercube class purity AUC | [0, 1] | Easy (high purity) |
| Neighbourhood Separability | NeighbourhoodSeparability | Same-class NN fraction AUC | [0, 1] | Easy (high separability) |

---

## Classical Measures

Dataset-level statistics that characterise the data distribution independent of class structure. See [classical/README.md](classical/README.md) for full equations and references.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Imbalance Ratio | IR | Majority-to-minority class size ratio | [1, ∞) | Imbalanced dataset |

---

## Distributional Measures

Statistical and geometric measures that directly compare class distributions or characterise the decision boundary. See [distributional/README.md](distributional/README.md) for full equations and references.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Silhouette Score | Silhouette | Cluster cohesion vs separation | [-1, 1] | Easy (well-separated) |
| Bhattacharyya Coefficient | Bhattacharyya | Histogram overlap between classes | [0, 1] | Hard (high overlap) |
| Wasserstein Distance | Wasserstein | Earth mover's distance between class distributions | [0, ∞) | Easy (large separation) |
| SVM Support Vector Ratio | SVM_SVR | Fraction of points needed to define the boundary | [0, 1] | Hard (complex boundary) |
| TwoNN Intrinsic Dimensionality | TwoNN_ID | Effective dimensionality of the data manifold | [1, n_features] | Complex geometry |

---

## Usage

```python
from data_complexity.data_metrics.metrics import ComplexityMetrics

dataset = {'X': X, 'y': y}
cm = ComplexityMetrics(dataset)

# All metrics as a flat dict of scalars
all_metrics = cm.get_all_metrics_scalar()

# By category (scalars — arrays aggregated via mean)
feature    = cm.feature_overlap_scalar()
instance   = cm.instance_overlap_scalar()
structural = cm.structural_overlap_scalar()
multi      = cm.multiresolution_overlap_scalar()
classical  = cm.classical_measures_scalar()
distrib    = cm.distributional_measures_scalar()

# Full (arrays returned as-is for per-class/per-feature detail)
feature_full = cm.feature_overlap_full()
multi_full   = cm.multiresolution_overlap_full()
```
