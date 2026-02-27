# Structural Overlap Measures

These measures capture the topology of the class boundary and the geometric structure of clusters, using graph-based and hypersphere-cover algorithms.

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

### N1 — Fraction of Borderline Points (MST)

Constructs the Minimum Spanning Tree (MST) over all samples using the pairwise distance matrix. Edges connecting samples of different classes are counted:

$$N1 = \frac{\text{count}(\text{MST edges linking different classes})}{n}$$

N1 estimates the density of the decision boundary: if classes are well separated, the MST will contain few inter-class edges.

Reference: Ho & Basu (2002). *Complexity measures of supervised classification problems.* IEEE TPAMI 24(3):289–300.

---

### T1 — Fraction of Covering Hyperspheres

Greedily constructs a set of hyperspheres to cover all training samples. Each sphere is centred on a sample and its radius is set to the distance to the nearest opposite-class sample (the "nearest enemy"). Covered samples are removed and the process repeats:

$$T1 = \frac{\text{number of hyperspheres}}{n}$$

Many small spheres are needed when the classes are interleaved, resulting in a high T1.

Reference: Lorena et al. (2019). *How Complex is your classification problem?* ACM Computing Surveys 52(5):1–34.

---

### Clust — Number of Clusters

For each sample, defines its *local set* as all same-class samples that are closer to it than its nearest enemy. Samples are then clustered by assigning each to the largest local set it belongs to:

$$\text{Clust} = \frac{\text{number of clusters}}{n}$$

A highly fragmented class (many isolated same-class pockets) will have a large Clust value.

Reference: Leyva et al. (2014). *A set of complexity measures designed for applying meta-learning to instance selection.* IEEE TKDE 27(2):354–367.

---

### N2 — Ratio of Intra/Inter Class Nearest Neighbour Distance

For each sample, computes the distance to its nearest same-class neighbour (intra) and its nearest different-class neighbour (inter). The ratio is then normalised:

$$r = \frac{\sum_i d_{\text{intra}}(i)}{\sum_i d_{\text{inter}}(i)}, \qquad N2 = \frac{r}{1 + r}$$

$N2 \to 0$ when intra-class distances are much smaller than inter-class distances (well-separated); $N2 \to 1$ when intra $\approx$ inter (overlapping).

Reference: Ho & Basu (2002).

---

### ONB — Overlap of Neighbourhood Balls

Applies a greedy sphere-covering algorithm independently per class. Each sphere is centred on the uncovered sample furthest from the class boundary, with its radius equal to the distance to its nearest enemy. The ONB score is the mean fraction of instances per sphere that belong to a different class:

$$ONB = \text{mean}_{\text{sphere}} \frac{n_{\text{other class in sphere}}}{n_{\text{total in sphere}}}$$

High values indicate spheres that are densely populated by both classes.

Reference: van der Walt & Barnard (2007). *Measures for the characterisation of pattern-recognition data sets.* PRASA 2007.

---

### LSCAvg — Local Set Average Cardinality

For each sample, the *local set* is all same-class samples closer than the nearest enemy:

$$LSC(\mathbf{x}_i) = \bigl|\{\mathbf{x}_j : \text{label}(\mathbf{x}_j) = \text{label}(\mathbf{x}_i),\ d(\mathbf{x}_i, \mathbf{x}_j) < d(\mathbf{x}_i, \text{nearest enemy})\}\bigr|$$

$$LSCAvg = 1 - \frac{\sum_i LSC(\mathbf{x}_i)}{n^2}$$

The subtraction from 1 is a normalisation so that larger average local sets give lower (easier) values. Small local sets indicate isolated samples near the class boundary.

Reference: Leyva et al. (2014).

---

### DBC — Decision Boundary Complexity

Uses the sphere-cover (ONB or T1) to define a reduced graph: each sphere becomes a node. An MST is then built over the sphere centres. DBC is the fraction of inter-class edges in this sphere-level MST:

$$DBC = \frac{\text{count}(\text{inter-class MST edges between spheres})}{n_{\text{spheres}}}$$

DBC captures the complexity of the boundary at the level of class regions rather than individual points.

Reference: van der Walt (2008). *Data measures that characterise classification problems.* PhD thesis, University of Pretoria.

---

### NSG — Number of Same-class Groups (Average Sphere Size)

Uses the sphere cover to measure how tightly each class groups together. NSG is the average number of training instances per covering sphere:

$$NSG = \frac{n_{\text{instances}}}{n_{\text{spheres}}}$$

Large NSG means each sphere covers many points — the class is compact and few spheres are needed. Small NSG indicates fragmented class regions.

Reference: van der Walt & Barnard (2007).

---

### ICSV — Intra-class Spatial Variability

Extends the sphere cover by computing per-sphere densities and measuring how variable those densities are:

$$\text{volume}(s_i) = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}\, r_i^d, \qquad \text{density}(s_i) = \frac{n_{\text{instances}}(s_i)}{\text{volume}(s_i)}$$

$$ICSV = \text{std}\bigl(\text{density}(s_i)\bigr)$$

High ICSV means sphere densities are highly non-uniform — some regions are dense and others sparse — indicating irregular, fragmented class structure.

Reference: van der Walt & Barnard (2007).
