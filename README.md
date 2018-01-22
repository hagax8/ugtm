# uGTM
Generative Topographic Mapping with Python.

GTM is a dimensionality reduction algorithm created by Bishop et al. (https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf)

This is a python implementation (using sklearn, d3 and numpy, in uGTM.py): I am debugging this right now, so it might not be optimal yet. The kernel version of the algorithm (kGTM) is also implemented. You can also generate regression or classification maps, or evaluate the predictive accuracy (classification) or RMSE/R2 (regression) in repeated cross-validation experiments - all the workflows were implemented in runGTM.py.

Ex 1: Load some labeled test data, make a GTM map for classification (for regression: --labeltype discrete): 

python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out --model GTM

(you can also add ids with the --ids option to visualize data point ids on your html map, or generate a t-SNE map instead with --model t-SNE or kernel GTM with --model kGTM)

Ex 2: With PCA pre-processing (usually better if less than 100 principal components, because of floating point precision limit) and missing data:

python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out --model GTM --pca --n_components 50 --missing

Ex 3: Run 5-fold cross-validation repeated 10 times on GTM classification model to select parameters (for regression: --labeltype continuous): 

python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out --model GTM --optimize

Ex 4: Run 5-fold cross-validation repeated 10 times on GTM classification model to select parameters (for regression: --labeltype continuous), and do the same with SVC (classification) or SVR (regression)):

python runGTM.py  --data csvlist.dat --labels csvlist.lbls  --labeltype discrete --output out --model GTM --optimize

- Héléna A. Gaspar, hgaspar.chemoinfo@gmail.com


