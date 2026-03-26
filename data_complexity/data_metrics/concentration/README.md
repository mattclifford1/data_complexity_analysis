# Concentration Measures
This is a work in progress. The general idea is to have 'deltas' style measures

## Deltas
The deltas method needs a projection, which for a data measure is not possible without some assumptions e.g. PCA or a classifier. Therefore some adjustments need to be made.

### Feature basis
The easy way to apply to data is to do deltas on a per feature basis.

### Use the full dimensionality
We could use the support of the ball in all dimensions. Two scenarios:
 - separable data: report the level at which delta for both classes touch
 - non-separable data:
  - report the level of overlap at given delta (gradient of level of overlap as delta increases?)
    - potentially normalise by the distance from the mean/bound of the mean to the bound or the bound of the other class?