# Concentration Measures
This is a work in progress. The general idea is to have 'deltas' style measures

## Deltas
The as is deltas method needs a projection, which for a data measure is not possible without some assumptions e.g. PCA or a classifier. 

### Feature basis
The easy way to apply to data is to do deltas on a per feature basis.

### No projection
We could use the support of the ball in all dimensions. Two scenarios:
 - separable data: report the level at which delta for both classes touch
 - non-separable data: 
  - report the level of overlap at given deltas? (gradient of level of overlap as delta increases?)
  - delta when one of the class hits the other's mean?