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

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Max Fisher's Discriminant Ratio | F1 | Best single linear discriminant | [0, 1] | Hard (normalised: 1 = worst) |
| Directional-vector Max Fisher's | F1v | Best linear projection | [0, 1] | Hard |
| Volume of Overlapping Region | F2 | Feature-space overlap fraction | [0, 1] | Hard |
| Max Individual Feature Efficiency | F3 | Best single feature discriminability | [0, 1] | Hard |
| Collective Feature Efficiency | F4 | Sequential feature elimination | [0, 1] | Hard |
| Input Noise | IN | Fraction of instances in cross-class domains | [0, 1] | Hard |

### F1 — Max Fisher's Discriminant Ratio

For each one-vs-one class pair and each feature `f`, the raw Fisher discriminant ratio is:

```
fisher_f = (μ1_f - μ2_f)² / (σ1_f² + σ2_f²)
```

The maximum over all features is then normalised so that higher complexity gives higher values:

```
F1 = 1 / (1 + max_f(fisher_f))
```

F1 = 0 indicates perfect linear separability on one feature; F1 = 1 means no discriminative feature. Values are averaged over all one-vs-one pairs for multiclass datasets.

Reference: Luengo et al. (2011). *Addressing data complexity for imbalanced data sets.* Soft Computing 15(10):1909–1936.

### F1v — Directional-vector Fisher's Discriminant Ratio

Finds the projection direction `d` that maximises the Fisher criterion across all features jointly. For a one-vs-one pair with scatter matrices `W` (within-class) and `B` (between-class):

```
d = W⁻¹ (μ1 - μ2)
F1v_raw = (dᵀ B d) / (dᵀ W d)
F1v = 1 / (1 + F1v_raw)
```

Where `B = (μ1 - μ2)(μ1 - μ2)ᵀ` and `W` is the pooled within-class covariance. Unlike F1, F1v considers all features simultaneously.

Reference: Lorena et al. (2019). *How Complex is your classification problem?* ACM Computing Surveys 52(5):1–34.

### F2 — Volume of Overlapping Region

For each feature `f` and one-vs-one class pair, the overlapping interval is:

```
overlap_f = max(0, min(max_c1_f, max_c2_f) - max(min_c1_f, min_c2_f))
range_f   = max(max_c1_f, max_c2_f) - min(min_c1_f, min_c2_f)
F2 = ∏_f  (overlap_f / range_f)
```

The product over features gives the fraction of the joint feature space covered by the overlap region. F2 = 0 means classes are linearly separable by at least one feature; F2 = 1 means complete overlap in all features.

Reference: Lorena et al. (2019).

### F3 — Maximum Individual Feature Efficiency

For each feature `f`, count the samples that fall inside the overlapping range of both classes. F3 is the fraction of the least-overlapping feature:

```
F3 = min_f( n_overlap_f / n_total )
```

Lower values indicate that at least one feature cleanly separates most samples.

Reference: Lorena et al. (2019).

### F4 — Collective Feature Efficiency

Iteratively applies F3: at each step, the feature with the smallest overlap is used to discard the non-overlapping samples; that feature is then removed and the process repeats on the remaining samples and features. F4 is the fraction of samples still overlapping after all features are exhausted:

```
F4 = n_remaining / n_total
```

This captures cases where no single feature is sufficient but a combination can separate the classes.

Reference: Lorena et al. (2019).

### IN — Input Noise

For each sample in class A, counts whether it falls within the feature-value domain (min–max range) of class B across all features. The measure is the fraction of such cross-domain instances:

```
IN = count(samples from c_i within domain of c_j) / (n · n_features)
```

Averaged over all one-vs-one pairs. A high IN indicates that many samples reside in regions "owned" by the opposing class.

Reference: van der Walt & Barnard (2007). *Measures for the characterisation of pattern-recognition data sets.* PRASA 2007.

---

## Instance Overlap

These measures assess difficulty at the instance level — how many training points are in overlapping regions or are surrounded by opposite-class neighbours.

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

### Raug — Augmented R Value

For each one-vs-one class pair, Raug measures how frequently samples from one class have a majority of k-NN from the other class, weighted by the imbalance ratio IR:

```
R(c_i → c_j) = fraction of c_i samples with > θ·k neighbours from c_j
Raug = (1 / (IR + 1)) · R(c_i→c_j) + (IR / (IR + 1)) · R(c_j→c_i)
```

where `θ` is a threshold (default 0.5) and IR = n_majority / n_minority. This gives greater weight to minority-class overlap.

Reference: Borsos et al. (2018). *Dealing with overlap and imbalance: a new metric and approach.* Pattern Analysis and Applications 21(2):381–395.

### deg_overlap — Degree of Overlap

For each sample, find its k nearest neighbours and count how many have a different class label. The degree of overlap is:

```
deg_overlap = count(samples with ≥1 different-class NN) / n
```

A straightforward fraction of instances that have at least one "enemy" in their neighbourhood.

Reference: Mercier et al. (2018). *Analysing the footprint of classifiers in overlapped and imbalanced contexts.* IDA 2018, pp. 200–212.

### N3 — Error Rate of the 1-Nearest Neighbour Classifier

Leave-one-out 1-NN classification error on the training set:

```
N3 = count(samples misclassified by their nearest same-training-set neighbour) / n
```

N3 directly estimates the irreducible error of the simplest non-parametric classifier. High N3 indicates dense class overlap.

Reference: Ho & Basu (2002). *Complexity measures of supervised classification problems.* IEEE TPAMI 24(3):289–300.

### SI — Separability Index

The complement of N3: fraction of samples correctly classified by their nearest neighbour:

```
SI = count(samples whose nearest neighbour has the same label) / n
```

SI = 1 means every point is closer to a same-class point than to any opposite-class point; SI = 0 means every point's nearest neighbour is from a different class.

Reference: Thornton (1998). *Separability is a learner's best friend.* 4th Neural Computation and Psychology Workshop.

### N4 — Non-linearity of the 1-NN Classifier

Creates an interpolated test set by generating synthetic points between randomly chosen pairs of same-class samples:

```
x_new = α·x_i + (1-α)·x_j,   α ~ Uniform(0,1),   label(x_i) = label(x_j)
```

N4 is then the 1-NN error rate when these interpolated points are classified using the original training set as the reference:

```
N4 = 1-NN error on synthetic set
```

This measures how non-linearly the decision boundary varies in regions between same-class samples.

Reference: Lorena et al. (2019).

### kDN — K-Disagreeing Neighbours

For each sample, the kDN score is the fraction of its k nearest neighbours that have a different class label:

```
kDN(x_i) = count(NN with different label) / k
kDN = mean_i(kDN(x_i))
```

This is a smooth generalisation of N3: rather than a binary misclassified/correct decision, it measures the degree of neighbourhood disagreement.

Reference: Smith et al. (2014). *An instance level analysis of data complexity.* Machine Learning 95(2):225–256.

### D3 — Disjunct Size

For each sample, counts how many of its k nearest neighbours belong to the same class. If the same-class count is less than k/2, the sample is considered part of a small disjunct (a minority group surrounded by the other class):

```
D3 = count(samples where same-class NN count < k/2) / n_per_class
```

Reported per class. Large D3 values indicate a fragmented, non-contiguous class distribution.

Reference: Sotoca et al. (2006). *A meta-learning framework for pattern classification by means of data complexity measures.* Inteligencia Artificial 10(29):31–38.

### CM — Class Complexity Measure

A threshold-based simplification of kDN: a sample is "complex" if more than half of its k neighbours disagree with its label (i.e. kDN > 0.5):

```
CM = count(samples where kDN(x_i) > 0.5) / n
```

CM specifically captures instances that would be misclassified by a majority-vote k-NN classifier.

Reference: Anwar et al. (2014). *Measurement of data complexity for classification problems with unbalanced data.* Statistical Analysis and Data Mining 7(3):194–211.

### Borderline — Borderline Examples

Inspired by the BORDERLINE-SMOTE definition. For each sample, find its 5 nearest neighbours and count how many `m` come from the other class:

```
safe:       m < 2   (well inside own class region)
borderline: 2 ≤ m ≤ 3   (near the decision boundary)
rare:       m = 4
noise/outlier: m = 5
```

The metric returns the fraction of samples classified as borderline (2 ≤ m ≤ 3):

```
Borderline = count(borderline samples) / n
```

Reference: Napierała et al. (2010). *Learning from imbalanced data in presence of noisy and borderline examples.* RSCTC 2010, pp. 158–167.

---

## Structural Overlap

These measures capture the topology of the class boundary and the geometric structure of clusters.

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

### N1 — Fraction of Borderline Points (MST)

Constructs the Minimum Spanning Tree (MST) over all samples using the pairwise distance matrix. Edges connecting samples of different classes are counted:

```
N1 = count(MST edges linking different classes) / n
```

N1 estimates the density of the decision boundary: if classes are well separated, the MST will contain few inter-class edges.

Reference: Ho & Basu (2002). *Complexity measures of supervised classification problems.* IEEE TPAMI 24(3):289–300.

### T1 — Fraction of Covering Hyperspheres

Greedily constructs a set of hyperspheres to cover all training samples. Each sphere is centred on a sample and its radius is set to the distance to the nearest opposite-class sample (the "nearest enemy"). Covered samples are removed and the process repeats:

```
T1 = number_of_hyperspheres / n
```

Many small spheres are needed when the classes are interleaved, resulting in a high T1.

Reference: Lorena et al. (2019).

### Clust — Number of Clusters

For each sample, defines its *local set* as all same-class samples that are closer to it than its nearest enemy. Samples are then clustered by assigning each to the largest local set it belongs to:

```
Clust = number_of_clusters / n
```

A highly fragmented class (many isolated same-class pockets) will have a large Clust value.

Reference: Leyva et al. (2014). *A set of complexity measures designed for applying meta-learning to instance selection.* IEEE TKDE 27(2):354–367.

### N2 — Ratio of Intra/Inter Class Nearest Neighbour Distance

For each sample, computes the distance to its nearest same-class neighbour (intra) and its nearest different-class neighbour (inter). The ratio is then normalised:

```
r = Σ_i d_intra(i) / Σ_i d_inter(i)
N2 = r / (1 + r)
```

N2 → 0 when intra-class distances are much smaller than inter-class distances (well-separated); N2 → 1 when intra ≈ inter (overlapping).

Reference: Ho & Basu (2002).

### ONB — Overlap of Neighbourhood Balls

Applies a greedy sphere-covering algorithm independently per class. Each sphere is centred on the uncovered sample furthest from the class boundary, with its radius equal to the distance to its nearest enemy. The ONB score is the mean fraction of instances per sphere that belong to a different class:

```
ONB = mean_sphere( n_other_class_in_sphere / n_total_in_sphere )
```

High values indicate spheres that are densely populated by both classes.

Reference: van der Walt & Barnard (2007). *Measures for the characterisation of pattern-recognition data sets.* PRASA 2007.

### LSCAvg — Local Set Average Cardinality

For each sample, the *local set* is all same-class samples closer than the nearest enemy. The Local Set Cardinality (LSC) is:

```
LSC(x_i) = |{x_j : label(x_j) = label(x_i),  d(x_i, x_j) < d(x_i, nearest_enemy)}|
LSCAvg = 1 - (Σ_i LSC(x_i)) / n²
```

The subtraction from 1 is a normalisation so that larger average local sets give lower (easier) values. Small local sets indicate isolated samples near the class boundary.

Reference: Leyva et al. (2014).

### DBC — Decision Boundary Complexity

Uses the sphere-cover (ONB or T1) to define a reduced graph: each sphere becomes a node. An MST is then built over the sphere centres. DBC is the fraction of inter-class edges in this sphere-level MST:

```
DBC = count(inter-class MST edges between spheres) / n_spheres
```

DBC captures the complexity of the boundary at the level of class regions rather than individual points.

Reference: van der Walt et al. (2008). *Data measures that characterise classification problems.* PhD thesis, University of Pretoria.

### NSG — Number of Same-class Groups (Average Sphere Size)

Uses the sphere cover to measure how tightly each class groups together. NSG is the average number of training instances per covering sphere:

```
NSG = n_instances / n_spheres
```

Large NSG means each sphere covers many points — the class is compact and few spheres are needed. Small NSG indicates fragmented class regions.

Reference: van der Walt & Barnard (2007).

### ICSV — Intra-class Spatial Variability

Extends the sphere cover by computing per-sphere densities and measuring how variable those densities are:

```
volume(sphere_i) = π^(d/2) / Γ(d/2 + 1) · r_i^d
density(sphere_i) = n_instances(sphere_i) / volume(sphere_i)
ICSV = std(density over all spheres)
```

High ICSV means sphere densities are highly non-uniform — some regions are dense and others sparse — indicating irregular, fragmented class structure.

Reference: van der Walt & Barnard (2007).

---

## Multiresolution Overlap

These measures analyse class purity across multiple spatial resolutions, from coarse to fine.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Multiresolution Class Aggregate | MRCA | Weighted impurity across resolutions | [0, 1] | Hard |
| Class Entropy | C1 | Mean neighbourhood class entropy | [0, 1] | Hard |
| Multi-resolution Complexity | C2 | Distance-weighted neighbourhood error | [0, 1] | Hard |
| Purity | Purity | Hypercube class purity AUC | [0, 1] | Easy (high purity) |
| Neighbourhood Separability | NeighbourhoodSeparability | Same-class NN fraction AUC | [0, 1] | Easy (high separability) |

### MRCA — Multiresolution Class Aggregate

For each one-vs-one class pair, builds *resolution profiles* by evaluating neighbourhood class compositions at multiple radii σ. For sample `x_i` in class 1:

```
ψ_i(σ) = (n_class2_in_σ(x_i) - n_class1_in_σ(x_i)) / (n_class2_in_σ(x_i) + n_class1_in_σ(x_i))
```

The profiles are clustered (k-means) and for each cluster, an MRI (Multiresolution Indicator) value is computed as a decreasing weighted sum over resolutions:

```
MRI = Σ_σ  w(σ) · (1 - profile(σ))
```

where weights decrease with resolution. MRCA returns one MRI value per cluster, reflecting the typical complexity experienced by groups of similar samples.

Reference: Armano & Tamponi (2016). *Experimenting multiresolution analysis for identifying regions of different classification complexity.* Pattern Analysis and Applications 19(1):129–137.

### C1 — Case Complexity Measure 1 (Neighbourhood Class Entropy)

For each sample `x_i` and neighbourhood size `k = 1…K`:

```
p_k(x_i) = count(same-class NN among k nearest) / k
C1(x_i) = 1 - (1/K) Σ_{k=1}^{K} p_k(x_i)
C1 = mean_i( C1(x_i) )
```

C1 measures how quickly the neighbourhood of each sample becomes "polluted" by the other class as the neighbourhood grows. High C1 means the neighbourhood is heterogeneous even at small k.

Reference: Massie et al. (2005). *Complexity-guided case discovery for case based reasoning.* AAAI 2005, pp. 216–221.

### C2 — Case Complexity Measure 2 (Distance-Weighted Complexity)

Similar to C1 but weights each neighbour's contribution by its normalised distance (so closer neighbours matter more):

```
contrib_k(x_i) = Σ_{j=1}^{k} max(0, 1 - d(x_i, x_j))  /  k
C2(x_i) = 1 - (1/K) Σ_{k=1}^{K} contrib_k(x_i)
C2 = mean_i( C2(x_i) )
```

Distances are normalised to [0, 1] before use. C2 penalises near-boundary instances more heavily because close opposite-class neighbours contribute larger values.

Reference: Massie et al. (2005).

### Purity

Partitions the feature space into a grid of hypercubes at multiple resolution levels `r = 0, 1, …, R`. At each resolution, cell purity is:

```
purity(cell) = sqrt( C/(C-1) · Σ_c (p_c - 1/C)² )
```

where `p_c` is the proportion of class `c` in the cell and C is the number of classes. This is the normalised standard deviation of class proportions — 0 for uniform mixing, 1 for pure cells. A weighted AUC is computed across resolutions (weight = `1/2^r`), then normalised by 0.702 (the maximum achievable AUC):

```
Purity = AUC(resolutions, weighted_purity) / 0.702
```

Reference: Singh (2003). *Prism: a novel framework for pattern recognition.* Pattern Analysis & Applications 6(2):134–149.

### NeighbourhoodSeparability

Also uses a multi-resolution hypercube grid. Within each cell at each resolution, for every sample `x_i` the proportion of same-class samples among its `k` nearest neighbours inside the cell is measured for `k = 1…K_cell`. The AUC of this proportion curve is computed per sample, then averaged, then weighted across resolutions:

```
sep(x_i, r) = AUC( k/K_cell,  same_class_NN_fraction(x_i, k) )
NS(r) = mean_i( sep(x_i, r) )
NeighbourhoodSeparability = AUC(resolutions, (1/2^r) · NS(r))
```

High values indicate that, at many resolution levels, samples are predominantly surrounded by same-class neighbours.

Reference: Singh (2003).

---

## Classical Measures

Dataset-level statistics that characterise the data distribution independent of class structure.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Imbalance Ratio | IR | Majority-to-minority class size ratio | [1, ∞) | Imbalanced dataset |

### IR — Imbalance Ratio

```
IR = n_majority / n_minority
```

where `n_majority` is the size of the largest class and `n_minority` the smallest. IR = 1 for a perfectly balanced dataset. For multiclass problems, the extreme class counts are used.

A high IR indicates that one class dominates the dataset, which can cause classifiers to be biased toward the majority class and inflates aggregate accuracy while missing the minority class.

---

## Distributional Measures

Statistical and geometric measures that directly compare class distributions or characterise the decision boundary.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Silhouette Score | Silhouette | Cluster cohesion vs separation | [-1, 1] | Easy (well-separated) |
| Bhattacharyya Coefficient | Bhattacharyya | Histogram overlap between classes | [0, 1] | Hard (high overlap) |
| Wasserstein Distance | Wasserstein | Earth mover's distance between class distributions | [0, ∞) | Easy (large separation) |
| SVM Support Vector Ratio | SVM_SVR | Fraction of points needed to define the boundary | [0, 1] | Hard (complex boundary) |
| TwoNN Intrinsic Dimensionality | TwoNN_ID | Effective dimensionality of the data manifold | [1, n_features] | Complex geometry |

### Silhouette Score

For each sample `i`:

```
a(i) = mean distance to all other samples in the same class
b(i) = mean distance to all samples in the nearest other class
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

The mean `s(i)` over all samples is the Silhouette score. Values near +1 indicate dense, well-separated clusters; values near 0 indicate overlapping clusters; negative values indicate misassignment.

Reference: Rousseeuw (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics 20:53–65.

### Bhattacharyya Coefficient

For two class distributions P and Q, estimated via equal-width histograms with K = 10 bins over the combined range of each feature:

```
BC(P, Q) = Σ_{k=1}^{K} √(p_k · q_k) · Δx
```

where `p_k` and `q_k` are the (density-normalised) histogram heights and `Δx` is the bin width. BC = 1 means identical distributions; BC = 0 means no overlap. This is averaged over all features and all pairwise class combinations.

Reference: Bhattacharyya (1943). *On a measure of divergence between two statistical populations defined by their probability distributions.* Bulletin of the Calcutta Mathematical Society 35:99–109.

### Wasserstein Distance

For each feature `f` and class pair `(c1, c2)`, the 1-D Wasserstein (Earth Mover's) distance is computed on scale-normalised samples:

```
W_f(c1, c2) = wasserstein_distance(X[y==c1, f] / σ_f,  X[y==c2, f] / σ_f)
```

where `σ_f` is the standard deviation of feature `f` across all samples. Division by `σ_f` makes the distance scale-invariant across features. The result is averaged over all features and class pairs. W = 0 means identical distributions; larger values indicate greater separation.

Reference: Villani (2003). *Topics in Optimal Transportation.* AMS Graduate Studies in Mathematics.

### SVM Support Vector Ratio

An RBF-kernel SVM (C = 1.0) is fitted on the full dataset. The support vector ratio is:

```
SVR = |support vectors| / n
```

Support vectors are the training points lying on or inside the margin. A high SVR indicates that many points are near the decision boundary, implying a complex, non-linear boundary or heavily overlapping classes. SVR = 0 is not achievable in practice (at least one SV per class is required); SVR approaching 1 means the entire training set is needed to define the boundary.

### TwoNN Intrinsic Dimensionality

The TwoNN estimator (Facco et al., 2017) exploits the ratio `μ_i = r2_i / r1_i`, where `r1_i` and `r2_i` are the distances to the 1st and 2nd nearest neighbours of point `i`. Under the assumption that data lies on a smooth manifold of intrinsic dimension `d`, the `μ_i` follow a Pareto distribution with parameter `d`:

```
P(μ > x) = x^{-d}
```

The maximum likelihood estimate is:

```
ID = (n - 1) / Σ_i log(μ_i)      (equivalent to -1 / mean(log(1/μ_i)))
```

This estimates the effective dimensionality of the data manifold, independent of the ambient feature space dimension. High ID means the data occupies a high-dimensional manifold, which generally makes classification harder.

Reference: Facco et al. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Nature Communications 8:14065.

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

---

## References

- Lorena et al. (2019). *How Complex is your classification problem? A survey on measuring classification complexity.* ACM Computing Surveys 52(5):1–34.
- Ho & Basu (2002). *Complexity measures of supervised classification problems.* IEEE TPAMI 24(3):289–300.
- Luengo et al. (2011). *Addressing data complexity for imbalanced data sets.* Soft Computing 15(10):1909–1936.
- Borsos et al. (2018). *Dealing with overlap and imbalance: a new metric and approach.* Pattern Analysis and Applications 21(2):381–395.
- Smith et al. (2014). *An instance level analysis of data complexity.* Machine Learning 95(2):225–256.
- Anwar et al. (2014). *Measurement of data complexity for classification problems with unbalanced data.* Statistical Analysis and Data Mining 7(3):194–211.
- Napierała et al. (2010). *Learning from imbalanced data in presence of noisy and borderline examples.* RSCTC 2010, pp. 158–167.
- Leyva et al. (2014). *A set of complexity measures designed for applying meta-learning to instance selection.* IEEE TKDE 27(2):354–367.
- van der Walt & Barnard (2007). *Measures for the characterisation of pattern-recognition data sets.* PRASA 2007.
- van der Walt (2008). *Data measures that characterise classification problems.* PhD thesis, University of Pretoria.
- Armano & Tamponi (2016). *Experimenting multiresolution analysis for identifying regions of different classification complexity.* Pattern Analysis and Applications 19(1):129–137.
- Massie et al. (2005). *Complexity-guided case discovery for case based reasoning.* AAAI 2005, pp. 216–221.
- Singh (2003). *Prism: a novel framework for pattern recognition.* Pattern Analysis & Applications 6(2):134–149.
- Mercier et al. (2018). *Analysing the footprint of classifiers in overlapped and imbalanced contexts.* IDA 2018, pp. 200–212.
- Sotoca et al. (2006). *A meta-learning framework for pattern classification by means of data complexity measures.* Inteligencia Artificial 10(29):31–38.
- Thornton (1998). *Separability is a learner's best friend.* 4th Neural Computation and Psychology Workshop.
- Rousseeuw (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics 20:53–65.
- Bhattacharyya (1943). *On a measure of divergence between two statistical populations.* Bulletin of the Calcutta Mathematical Society 35:99–109.
- Facco et al. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Nature Communications 8:14065.
- Villani (2003). *Topics in Optimal Transportation.* AMS Graduate Studies in Mathematics.
