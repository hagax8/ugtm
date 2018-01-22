# uGTM v1.0 beta: Generative Topographic Mapping with Python.

Generative Topographic Maps (GTMs) are probabilistic Kohonen maps. GTM is a dimensionality reduction algorithm (as t-SNE, LLE, etc).

This is a python implementation of GTMs, using sklearn, d3 and numpy: I am debugging this right now, so it might not be optimal yet. 

uGTM.py contains the core functions and runGTM.py is an easy-to-use program. The kernel version of the algorithm (kGTM) is also implemented. You can also generate regression or classification maps, or evaluate the predictive accuracy (classification) or RMSE/R2 (regression) in repeated cross-validation experiments - all the workflows were implemented in runGTM.py, which uses the uGTM.py core functions.

GTM is a dimensionality reduction algorithm created by Bishop et al. (https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf)


## Prerequisites
python 2.7

and following packages:
sklearn
numpy 
matplotlib
pandas
scipy
mpl_toolkits

## Examples

### Ex 1: Load some labeled test data, make a GTM map for classification (for regression: --labeltype discrete): 

```
python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out --model GTM
```
This will generate a png and an html map. For GTM, there are two possible representations: the mean position, and the mode (the position where the data point has the highest posterior probability). On the html map, each point represents the mean position, and a grey line shows which node on the 2D map corresponds to the mode.

Instead of --data and --labels, you can load some test data from sklearn using --usetest (s = S in 3D space, swiss = swiss roll in 3D space, iris = iris classification dataset):

```
python runGTM.py  --usetest s --labeltype continuous --output out --model compare
```

```
python runGTM.py  --usetest iris --labeltype discrete --output out --model GTM
```

(you can also add ids with the --ids option to visualize data point ids on your html map, or generate a t-SNE map instead with --model t-SNE or kernel GTM with --model kGTM)

### Ex 2: With PCA pre-processing (usually better if less than 100 principal components, because of floating point precision limit) and missing data:

```
python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out \
--model GTM --pca --n_components  50 --missing --missing_strategy median
```

### Ex 3: Run 5-fold cross-validation repeated 10 times on GTM classification model to select parameters (for regression: --labeltype continuous): 

```
python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out \
--model GTM --optimize
```


### Ex 4: Run 5-fold cross-validation repeated 10 times on GTM classification model to select parameters (for regression: --labeltype continuous), and do the same with SVC (classification) or SVR (regression)):

```
python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out \
--model compare --optimize
```

## Get help

```
python runGTM.py -h
```


## Author

Héléna A. Gaspar, hgaspar.chemoinfo@gmail.com, https://github.com/hagax8


