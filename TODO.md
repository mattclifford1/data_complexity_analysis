## change cross val to use non sklearn models ()
dont rely on _create_model()


## ml metrics
change the [0] hack for matrix metrics to return scalars


## experiments
 - plot full dataset and train/test splits

## pycol
 - make sure we are doing the metrics on the train data 
 - would doing the metrics on the test data be more informative? if there is a difference between the train and test metrics, that could be interesting to look at.