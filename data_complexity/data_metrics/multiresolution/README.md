# Multiresolution Overlap Measures

These measures analyse class purity across multiple spatial resolutions, from coarse to fine, using neighbourhood graphs and hypercube grids.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Multiresolution Class Aggregate | MRCA | Weighted impurity across resolutions | [0, 1] | Hard |
| Class Entropy | C1 | Mean neighbourhood class entropy | [0, 1] | Hard |
| Multi-resolution Complexity | C2 | Distance-weighted neighbourhood error | [0, 1] | Hard |
| Purity | Purity | Hypercube class purity AUC | [0, 1] | Easy (high purity) |
| Neighbourhood Separability | NeighbourhoodSeparability | Same-class NN fraction AUC | [0, 1] | Easy (high separability) |

---

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

---

### C1 — Case Complexity Measure 1 (Neighbourhood Class Entropy)

For each sample `x_i` and neighbourhood size `k = 1…K`:

```
p_k(x_i) = count(same-class NN among k nearest) / k
C1(x_i) = 1 - (1/K) Σ_{k=1}^{K} p_k(x_i)
C1 = mean_i( C1(x_i) )
```

C1 measures how quickly the neighbourhood of each sample becomes "polluted" by the other class as the neighbourhood grows. High C1 means the neighbourhood is heterogeneous even at small k.

Reference: Massie et al. (2005). *Complexity-guided case discovery for case based reasoning.* AAAI 2005, pp. 216–221.

---

### C2 — Case Complexity Measure 2 (Distance-Weighted Complexity)

Similar to C1 but weights each neighbour's contribution by its normalised distance (so closer neighbours matter more):

```
contrib_k(x_i) = Σ_{j=1}^{k} max(0, 1 - d(x_i, x_j))  /  k
C2(x_i) = 1 - (1/K) Σ_{k=1}^{K} contrib_k(x_i)
C2 = mean_i( C2(x_i) )
```

Distances are normalised to [0, 1] before use. C2 penalises near-boundary instances more heavily because close opposite-class neighbours contribute larger values.

Reference: Massie et al. (2005).

---

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

---

### NeighbourhoodSeparability

Also uses a multi-resolution hypercube grid. Within each cell at each resolution, for every sample `x_i` the proportion of same-class samples among its `k` nearest neighbours inside the cell is measured for `k = 1…K_cell`. The AUC of this proportion curve is computed per sample, then averaged, then weighted across resolutions:

```
sep(x_i, r) = AUC( k/K_cell,  same_class_NN_fraction(x_i, k) )
NS(r) = mean_i( sep(x_i, r) )
NeighbourhoodSeparability = AUC(resolutions, (1/2^r) · NS(r))
```

High values indicate that, at many resolution levels, samples are predominantly surrounded by same-class neighbours.

Reference: Singh (2003).
