# Feature Overlap Measures

These measures assess how well individual features or linear combinations of features can separate classes. All use a one-vs-one strategy for multiclass datasets.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Max Fisher's Discriminant Ratio | F1 | Best single linear discriminant | [0, 1] | Hard (normalised: 1 = worst) |
| Directional-vector Max Fisher's | F1v | Best linear projection | [0, 1] | Hard |
| Volume of Overlapping Region | F2 | Feature-space overlap fraction | [0, 1] | Hard |
| Max Individual Feature Efficiency | F3 | Best single feature discriminability | [0, 1] | Hard |
| Collective Feature Efficiency | F4 | Sequential feature elimination | [0, 1] | Hard |
| Input Noise | IN | Fraction of instances in cross-class domains | [0, 1] | Hard |

---

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

---

### F1v — Directional-vector Fisher's Discriminant Ratio

Finds the projection direction `d` that maximises the Fisher criterion across all features jointly. For a one-vs-one pair with scatter matrices `W` (within-class) and `B` (between-class):

```
d = W⁻¹ (μ1 - μ2)
F1v_raw = (dᵀ B d) / (dᵀ W d)
F1v = 1 / (1 + F1v_raw)
```

Where `B = (μ1 - μ2)(μ1 - μ2)ᵀ` and `W` is the pooled within-class covariance. Unlike F1, F1v considers all features simultaneously.

Reference: Lorena et al. (2019). *How Complex is your classification problem?* ACM Computing Surveys 52(5):1–34.

---

### F2 — Volume of Overlapping Region

For each feature `f` and one-vs-one class pair, the overlapping interval is:

```
overlap_f = max(0, min(max_c1_f, max_c2_f) - max(min_c1_f, min_c2_f))
range_f   = max(max_c1_f, max_c2_f) - min(min_c1_f, min_c2_f)
F2 = ∏_f  (overlap_f / range_f)
```

The product over features gives the fraction of the joint feature space covered by the overlap region. F2 = 0 means classes are linearly separable by at least one feature; F2 = 1 means complete overlap in all features.

Reference: Lorena et al. (2019).

---

### F3 — Maximum Individual Feature Efficiency

For each feature `f`, count the samples that fall inside the overlapping range of both classes. F3 is the fraction at the least-overlapping feature:

```
F3 = min_f( n_overlap_f / n_total )
```

Lower values indicate that at least one feature cleanly separates most samples.

Reference: Lorena et al. (2019).

---

### F4 — Collective Feature Efficiency

Iteratively applies F3: at each step, the feature with the smallest overlap is used to discard the non-overlapping samples; that feature is then removed and the process repeats on the remaining samples and features. F4 is the fraction of samples still overlapping after all features are exhausted:

```
F4 = n_remaining / n_total
```

This captures cases where no single feature is sufficient but a combination can separate the classes.

Reference: Lorena et al. (2019).

---

### IN — Input Noise

For each sample in class A, counts whether it falls within the feature-value domain (min–max range) of class B across all features. The measure is the fraction of such cross-domain instances:

```
IN = count(samples from c_i within domain of c_j) / (n · n_features)
```

Averaged over all one-vs-one pairs. A high IN indicates that many samples reside in regions "owned" by the opposing class.

Reference: van der Walt & Barnard (2007). *Measures for the characterisation of pattern-recognition data sets.* PRASA 2007.
