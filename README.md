# ugtm: Generative Topographic Mapping with Python.

Link to the package documentation: http://ugtm.readthedocs.io/en/latest/

GTM (Generative Topographic Mapping) is a dimensionality reduction algorithm (as t-SNE, LLE, etc) created by Bishop et al. (https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf) and a probabilistic counterpart of Kohonen maps.

ugtm is a python package implementing GTM and GTM prediction algorithms. ugtm contains the core functions and runGTM.py (in bin directory) is an easy-to-use program. The kernel version of the algorithm (kGTM) is also implemented. You can also generate regression or classification maps, or evaluate the predictive accuracy (classification) or RMSE/R2 (regression) in repeated cross-validation experiments.

## Install ugtm

Simple installation:
- pip install ugtm

If you get error messages, try upgrading the following packages:
- pip install --upgrade pip numpy scikit-learn matplotlib scipy mpld3 jinja2
OR:
- sudo pip install --upgrade pip numpy scikit-learn matplotlib scipy mpld3 jinja2

If you have problems with anaconda packages, try to create a virtual env called "p2" for python 2.7.14:
- conda create -n p2 python=2.7.14 numpy=1.14.5 scikit-learn=0.19 matplotlib=2.2.2 scipy=0.19.1 mpld3=0.3 jinja2=2.10
- source activate p2
- pip install ugtm

Or p3 for python 3.6.6:
- conda create -n p3 python=3.6.6 numpy=1.14.5 scikit-learn=0.19 matplotlib=2.2.2 scipy=0.19.1 mpld3=0.3 jinja2=2.10
- source activate p3
- pip install ugtm


## Documentation

[Readthedocs](http://ugtm.readthedocs.io/en/latest/)


## Prerequisites

Python 2.7 or + (tested on Python 3.4.6 and Python 2.7.14)

and following packages:
- scikit-learn>=0.19
- numpy>=1.14.5
- matplotlib>=2.2.2
- scipy>=0.19.1
- mpld3>=0.3
- jinja2>=2.10


## Author

Héléna A. Gaspar, hagax8@gmail.com, https://github.com/hagax8


