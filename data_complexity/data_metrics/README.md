# Data Complexity Metrics

This module implements complexity measures for classification datasets, grouped into six categories. Each measure quantifies a different aspect of how difficult a dataset is to classify.

---

## Categories

| Category | Count | What it measures |
|---|---|---|
| [Feature Overlap](#feature-overlap) | 6 | Discriminability of individual features |
| [Instance Overlap](#instance-overlap) | 9 | Hardness of individual training instances |
| [Structural Overlap](#structural-overlap) | 9 | Topology of the class boundary |
| [Multiresolution Overlap](#multiresolution-overlap) | 5 | Class purity at multiple resolutions |
| [Classical Measures](#classical-measures) | 1 | Dataset-level statistics |
| [Distributional Measures](#distributional-measures) | 5 | Statistical distribution overlap and boundary geometry |

---

## Feature Overlap

These measures assess how well individual features or linear combinations of features can separate classes.

| Name | Symbol | Intuition | Equation | Range | High value means |
|---|---|---|---|---|---|
| Max Fisher's Discriminant Ratio | F1 | Best single linear discriminant | `F1 = max_f (μ1_f - μ2_f)² / (σ1_f² + σ2_f²)` | [0, ∞) | Easy (well-separated) |
| Directional-vector Max Fisher's | F1v | Best linear projection (multiclass) | Fisher criterion on optimised direction vector | [0, ∞) | Easy |
| Volume of Overlapping Region | F2 | Fraction of feature-space overlap | Product of per-feature overlap ratios | [0, 1] | Hard (high overlap) |
| Max Individual Feature Efficiency | F3 | Best single feature discriminability | `F3 = (n - n_overlap) / n` for best feature | [0, 1] | Easy |
| Collective Feature Efficiency | F4 | Sequential elimination of useful features | Fraction of points separable after removing discriminating features | [0, 1] | Hard |
| Input Noise | IN | Sensitivity of classification to feature perturbations | Fraction of instances that change class when features are slightly perturbed | [0, 1] | Hard |

---

## Instance Overlap

These measures assess difficulty at the instance level — how many training points are in overlapping regions or are surrounded by opposite-class neighbours.

| Name | Symbol | Intuition | Equation | Range | High value means |
|---|---|---|---|---|---|
| Augmented R Value | Raug | Mean instance hardness | Average fraction of k-NN from opposite class | [0, 1] | Hard |
| Degree of Overlap | deg_overlap | Direct overlap count | Fraction of instances within the overlap hypercube | [0, 1] | Hard |
| Error Rate of 1-NN | N3 | Leave-one-out 1-NN error | LOO-1NN misclassification rate | [0, 1] | Hard |
| Separability Index | SI | Local separability | 1 - fraction of instances with same-class nearest neighbour | [0, 1] | Hard |
| Non-linearity of 1-NN | N4 | Difficulty of interpolating the boundary | Error rate of 1-NN on linear interpolation of training set | [0, 1] | Hard |
| K-Disagreeing Neighbours | kDN | Neighbourhood disagreement | Mean fraction of k-NN with different class label | [0, 1] | Hard |
| Disjunct Size | D3 | Fragmentation of decision regions | Mean normalised size of disjuncts in a decision tree | [0, 1] | Hard (many small disjuncts) |
| Class Complexity Measure | CM | Tree-based complexity | Weighted mean depth of leaves in a decision tree | [0, ∞) | Hard |
| Borderline Examples | Borderline | Fraction of points near the boundary | Fraction of instances that are borderline (between-class kNN) | [0, 1] | Hard |

---

## Structural Overlap

These measures capture the topology of the class boundary and the geometric structure of clusters.

| Name | Symbol | Intuition | Equation | Range | High value means |
|---|---|---|---|---|---|
| Fraction of Borderline Points | N1 | Boundary density | Fraction of samples in the minimum spanning tree that connect classes | [0, 1] | Hard |
| Fraction of Covering Hyperspheres | T1 | Boundary complexity | Fraction of hyperspheres needed to cover data without overlap | [0, 1] | Hard |
| Number of Clusters | Clust | Class fragmentation | Mean number of clusters per class (normalised) | [0, ∞) | Hard |
| Intra/Inter NN Distance Ratio | N2 | Class compactness | `N2 = Σ d_intra / Σ d_inter` over nearest neighbours | [0, ∞) | Hard (high intra, low inter) |
| Overlap of Neighbourhoods | ONB | Neighbourhood class mixing | Mean fraction of neighbourhood occupied by other classes | [0, 1] | Hard |
| Local Set Average Cardinality | LSCAvg | Local set size | Average size of the local set (points closer than the nearest opposite-class neighbour) | [1, n] | Easy (large local sets) |
| Decision Boundary Complexity | DBC | Boundary density via hyperspheres | Fraction of hyperspheres that contain points from more than one class | [0, 1] | Hard |
| Number of Same-class Groups | NSG | Class dispersion | Number of connected same-class groups per hypersphere cover | [0, ∞) | Hard |
| Intra-class Spatial Variability | ICSV | Within-class spread | Normalised variance of intra-class distances | [0, 1] | Hard |

---

## Multiresolution Overlap

These measures analyse class purity across multiple spatial resolutions, from coarse to fine.

| Name | Symbol | Intuition | Equation | Range | High value means |
|---|---|---|---|---|---|
| Multiresolution Class Aggregate | MRCA | Mean class purity across resolutions | Weighted average impurity over resolution levels | [0, 1] | Hard |
| Class Entropy (C1) | C1 | Class proportion entropy | `C1 = -Σ p_c log(p_c)` normalised | [0, 1] | Hard (balanced/many classes) |
| Multi-resolution Complexity | C2 | Boundary complexity over k-NN graphs | Mean misclassification rate over increasing k | [0, 1] | Hard |
| Purity | Purity | Cluster purity | Mean fraction of dominant class per neighbourhood | [0, 1] | Easy (high purity) |
| Neighbourhood Separability | NeighbourhoodSeparability | Separability at multiple neighbourhood sizes | Mean fraction of same-class neighbours across resolutions | [0, 1] | Easy (high separability) |

---

## Classical Measures

Dataset-level statistics that characterise the data distribution independent of class structure.

| Name | Symbol | Intuition | Equation | Range | High value means |
|---|---|---|---|---|---|
| Imbalance Ratio | IR | Class imbalance | `IR = n_majority / n_minority` | [1, ∞) | Imbalanced dataset |

---

## Distributional Measures

Statistical and geometric measures that directly compare class distributions or characterise the decision boundary.

| Name | Symbol | Intuition | Equation | Range | High value means |
|---|---|---|---|---|---|
| Silhouette Score | Silhouette | Cluster cohesion vs separation | `s(i) = (b(i) - a(i)) / max(a(i), b(i))`, averaged | [-1, 1] | Easy (well-separated, high score) |
| Bhattacharyya Coefficient | Bhattacharyya | Histogram overlap between classes | `BC(p,q) = Σ √(p_i · q_i) · Δx` averaged per feature and class pair | [0, 1] | Hard (high overlap) |
| Wasserstein Distance | Wasserstein | Earth mover's distance between class distributions | `W(P,Q) = inf_γ E[‖X-Y‖]`, averaged per feature/pair, scale-normalised | [0, ∞) | Easy (large separation) |
| SVM Support Vector Ratio | SVM_SVR | Fraction of points needed to define the boundary | `SVR = n_support_vectors / n_samples` for RBF-SVM | [0, 1] | Hard (complex boundary) |
| TwoNN Intrinsic Dimensionality | TwoNN_ID | Effective dimensionality of the data manifold | TwoNN estimator (Facco et al. 2017): `ID = -1 / mean(log(r1/r2))` | [1, n_features] | Complex geometry |

### Silhouette Score detail

For sample `i`:
```
a(i) = mean distance to same-class samples
b(i) = mean distance to nearest other-class cluster
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Mean `s(i)` over all samples. Values near +1 indicate dense, well-separated clusters.

### Bhattacharyya Coefficient detail

For two class distributions P and Q estimated via histograms with K bins:
```
BC(P, Q) = sum_{k=1}^{K} sqrt(p_k * q_k) * bin_width
```
BC = 1 means identical distributions; BC = 0 means no overlap.
This is averaged over all features and all pairwise class combinations.

### Wasserstein Distance detail

For each feature `f` and each pair of classes `(c1, c2)`:
```
W_f(c1, c2) = wasserstein_distance(X[y==c1, f] / std_f, X[y==c2, f] / std_f)
```
Averaged over all features and class pairs. Division by `std_f` makes the distance scale-invariant.

### SVM Support Vector Ratio detail

An RBF-kernel SVM (C=1) is fitted on the dataset. The support vector ratio is:
```
SVR = |support vectors| / n_samples
```
A high SVR indicates that many points lie near the decision boundary, i.e. the boundary is complex or the classes are interleaved.

### TwoNN Intrinsic Dimensionality detail

The TwoNN estimator (Facco et al., 2017, *Nature Communications*) computes:
```
mu_i = r2_i / r1_i    (ratio of 2nd to 1st nearest-neighbour distance)
ID = -1 / mean(log(1 / mu_i))    (MLE of Pareto parameter)
```
where `r1_i` and `r2_i` are the distances to the 1st and 2nd nearest neighbours of point `i`. This estimates the intrinsic dimension of the data manifold, independent of the ambient dimension.

---

## Usage

```python
from data_complexity.data_metrics.metrics import ComplexityMetrics
import numpy as np

dataset = {'X': X, 'y': y}
cm = ComplexityMetrics(dataset)

# All metrics in one dict
all_metrics = cm.get_all_metrics_scalar()

# By category
feature   = cm.feature_overlap_scalar()
instance  = cm.instance_overlap_scalar()
structural = cm.structural_overlap_scalar()
multi     = cm.multiresolution_overlap_scalar()
classical = cm.classical_measures_scalar()
distrib   = cm.distributional_measures_scalar()
```

## References

- Lorena et al. (2019). *How Complex is your classification problem? A survey on measuring classification complexity.* ACM Computing Surveys.
- Facco et al. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Nature Communications.
- Bhattacharyya (1943). *On a measure of divergence between two statistical populations defined by their probability distributions.* Bulletin of the Calcutta Mathematical Society.
