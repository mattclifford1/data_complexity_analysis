# Distributional Measures

Statistical and geometric measures that directly compare class distributions or characterise the decision boundary. Unlike the PyCol-backed measures, these are implemented directly using sklearn, scipy, and skdim.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Silhouette Score | Silhouette | Cluster cohesion vs separation | [-1, 1] | Easy (well-separated) |
| Bhattacharyya Coefficient | Bhattacharyya | Histogram overlap between classes | [0, 1] | Hard (high overlap) |
| Wasserstein Distance | Wasserstein | Earth mover's distance between class distributions | [0, ∞) | Easy (large separation) |
| SVM Support Vector Ratio | SVM_SVR | Fraction of points needed to define the boundary | [0, 1] | Hard (complex boundary) |
| TwoNN Intrinsic Dimensionality | TwoNN_ID | Effective dimensionality of the data manifold | [1, n_features] | Complex geometry |

---

### Silhouette Score

For each sample $i$:

$$a(i) = \text{mean distance to all other samples in the same class}$$

$$b(i) = \text{mean distance to all samples in the nearest other class}$$

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

The mean $s(i)$ over all samples is the Silhouette score. Values near $+1$ indicate dense, well-separated clusters; values near $0$ indicate overlapping clusters; negative values indicate misassignment.

Reference: Rousseeuw (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics 20:53–65.

---

### Bhattacharyya Coefficient

For two class distributions $P$ and $Q$, estimated via equal-width histograms with $K = 10$ bins over the combined range of each feature:

$$BC(P, Q) = \sum_{k=1}^{K} \sqrt{p_k \cdot q_k} \cdot \Delta x$$

where $p_k$ and $q_k$ are the density-normalised histogram heights and $\Delta x$ is the bin width. $BC = 1$ means identical distributions; $BC = 0$ means no overlap. This is averaged over all features and all pairwise class combinations.

Reference: Bhattacharyya (1943). *On a measure of divergence between two statistical populations defined by their probability distributions.* Bulletin of the Calcutta Mathematical Society 35:99–109.

---

### Wasserstein Distance

For each feature $f$ and class pair $(c_1, c_2)$, the 1-D Wasserstein (Earth Mover's) distance is computed on scale-normalised samples:

$$W_f(c_1, c_2) = \mathcal{W}\!\left(\frac{X_{y=c_1,f}}{\sigma_f},\ \frac{X_{y=c_2,f}}{\sigma_f}\right)$$

where $\sigma_f$ is the standard deviation of feature $f$ across all samples. Division by $\sigma_f$ makes the distance scale-invariant. The result is averaged over all features and class pairs. $W = 0$ means identical distributions; larger values indicate greater separation.

Reference: Villani (2003). *Topics in Optimal Transportation.* AMS Graduate Studies in Mathematics.

---

### SVM Support Vector Ratio

An RBF-kernel SVM ($C = 1.0$) is fitted on the full dataset. The support vector ratio is:

$$SVR = \frac{|\text{support vectors}|}{n}$$

Support vectors are the training points lying on or inside the margin. A high SVR indicates that many points are near the decision boundary, implying a complex, non-linear boundary or heavily overlapping classes. $SVR$ approaching 1 means the entire training set is needed to define the boundary.

---

### TwoNN Intrinsic Dimensionality

The TwoNN estimator (Facco et al., 2017) exploits the ratio $\mu_i = r_{2,i} / r_{1,i}$, where $r_{1,i}$ and $r_{2,i}$ are the distances to the 1st and 2nd nearest neighbours of point $i$. Under the assumption that data lies on a smooth manifold of intrinsic dimension $d$, the $\mu_i$ follow a Pareto distribution:

$$P(\mu > x) = x^{-d}$$

The maximum likelihood estimate is:

$$ID = \frac{n - 1}{\sum_i \log \mu_i}$$

This estimates the effective dimensionality of the data manifold, independent of the ambient feature space dimension. High $ID$ means the data occupies a high-dimensional manifold, which generally makes classification harder.

Reference: Facco et al. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Nature Communications 8:14065.
