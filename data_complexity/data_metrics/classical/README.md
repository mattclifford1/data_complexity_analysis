# Classical Measures

Dataset-level statistics that characterise the data distribution independent of class structure.

| Name | Symbol | Intuition | Range | High value means |
|---|---|---|---|---|
| Imbalance Ratio | IR | Majority-to-minority class size ratio | [1, ∞) | Imbalanced dataset |

---

### IR — Imbalance Ratio

```
IR = n_majority / n_minority
```

where `n_majority` is the size of the largest class and `n_minority` the smallest. IR = 1 for a perfectly balanced dataset. For multiclass problems, the extreme class counts are used.

A high IR indicates that one class dominates the dataset, which can cause classifiers to be biased toward the majority class and inflates aggregate accuracy while missing the minority class.
