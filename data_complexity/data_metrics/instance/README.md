# Instance Overlap Measures

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

---

### Raug — Augmented R Value

For each one-vs-one class pair, Raug measures how frequently samples from one class have a majority of k-NN from the other class, weighted by the imbalance ratio IR:

```
R(c_i → c_j) = fraction of c_i samples with > θ·k neighbours from c_j
Raug = (1 / (IR + 1)) · R(c_i→c_j) + (IR / (IR + 1)) · R(c_j→c_i)
```

where `θ` is a threshold (default 0.5) and IR = n_majority / n_minority. This gives greater weight to minority-class overlap.

Reference: Borsos et al. (2018). *Dealing with overlap and imbalance: a new metric and approach.* Pattern Analysis and Applications 21(2):381–395.

---

### deg_overlap — Degree of Overlap

For each sample, find its k nearest neighbours and count how many have a different class label. The degree of overlap is:

```
deg_overlap = count(samples with ≥1 different-class NN) / n
```

A straightforward fraction of instances that have at least one "enemy" in their neighbourhood.

Reference: Mercier et al. (2018). *Analysing the footprint of classifiers in overlapped and imbalanced contexts.* IDA 2018, pp. 200–212.

---

### N3 — Error Rate of the 1-Nearest Neighbour Classifier

Leave-one-out 1-NN classification error on the training set:

```
N3 = count(samples misclassified by their nearest same-training-set neighbour) / n
```

N3 directly estimates the irreducible error of the simplest non-parametric classifier. High N3 indicates dense class overlap.

Reference: Ho & Basu (2002). *Complexity measures of supervised classification problems.* IEEE TPAMI 24(3):289–300.

---

### SI — Separability Index

The complement of N3: fraction of samples correctly classified by their nearest neighbour:

```
SI = count(samples whose nearest neighbour has the same label) / n
```

SI = 1 means every point is closer to a same-class point than to any opposite-class point; SI = 0 means every point's nearest neighbour is from a different class.

Reference: Thornton (1998). *Separability is a learner's best friend.* 4th Neural Computation and Psychology Workshop.

---

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

Reference: Lorena et al. (2019). *How Complex is your classification problem?* ACM Computing Surveys 52(5):1–34.

---

### kDN — K-Disagreeing Neighbours

For each sample, the kDN score is the fraction of its k nearest neighbours that have a different class label:

```
kDN(x_i) = count(NN with different label) / k
kDN = mean_i(kDN(x_i))
```

This is a smooth generalisation of N3: rather than a binary misclassified/correct decision, it measures the degree of neighbourhood disagreement.

Reference: Smith et al. (2014). *An instance level analysis of data complexity.* Machine Learning 95(2):225–256.

---

### D3 — Disjunct Size

For each sample, counts how many of its k nearest neighbours belong to the same class. If the same-class count is less than k/2, the sample is considered part of a small disjunct (a minority group surrounded by the other class):

```
D3 = count(samples where same-class NN count < k/2) / n_per_class
```

Reported per class. Large D3 values indicate a fragmented, non-contiguous class distribution.

Reference: Sotoca et al. (2006). *A meta-learning framework for pattern classification by means of data complexity measures.* Inteligencia Artificial 10(29):31–38.

---

### CM — Class Complexity Measure

A threshold-based simplification of kDN: a sample is "complex" if more than half of its k neighbours disagree with its label (i.e. kDN > 0.5):

```
CM = count(samples where kDN(x_i) > 0.5) / n
```

CM specifically captures instances that would be misclassified by a majority-vote k-NN classifier.

Reference: Anwar et al. (2014). *Measurement of data complexity for classification problems with unbalanced data.* Statistical Analysis and Data Mining 7(3):194–211.

---

### Borderline — Borderline Examples

Inspired by the BORDERLINE-SMOTE definition. For each sample, find its 5 nearest neighbours and count how many `m` come from the other class:

```
safe:          m < 2   (well inside own class region)
borderline:    2 ≤ m ≤ 3   (near the decision boundary)
rare:          m = 4
noise/outlier: m = 5
```

The metric returns the fraction of samples classified as borderline (2 ≤ m ≤ 3):

```
Borderline = count(borderline samples) / n
```

Reference: Napierała et al. (2010). *Learning from imbalanced data in presence of noisy and borderline examples.* RSCTC 2010, pp. 158–167.
