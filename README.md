# ugtm: Generative Topographic Mapping with Python

GTM (Generative Topographic Mapping) is a dimensionality reduction algorithm (like t-SNE, LLE, etc.) created by Bishop et al. ([paper](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf)) and a probabilistic counterpart of Kohonen maps.

ugtm implements GTM and GTM-based prediction algorithms, including a kernel variant (kGTM), classification (GTC) and regression (GTR) maps, sklearn-compatible estimators, and repeated cross-validation.

## Documentation

Full documentation including examples: [ugtm.readthedocs.io](https://ugtm.readthedocs.io/)

## Install

```bash
pip install ugtm
```

Requires Python >= 3.8 and:
- numpy >= 1.21
- scikit-learn >= 1.0
- scipy >= 1.7
- jinja2 >= 3.0

## Quick start

```python
import ugtm
import numpy as np

data = np.random.randn(100, 50)
labels = np.random.choice([1, 2], size=100)

# Fit a GTM map
gtm = ugtm.runGTM(data=data)

# Access 2D representations
coordinates = gtm.matMeans          # mean positions (n_samples, 2)
modes       = gtm.matModes          # mode positions (n_samples, 2)
resp        = gtm.matR              # responsibilities (n_samples, n_nodes)
```

### sklearn-compatible estimators

```python
from ugtm import eGTM, eGTC, eGTR

X_train = np.random.randn(100, 50)
X_test  = np.random.randn(50, 50)
y_train = np.random.choice([1, 2, 3], size=100)

transformed      = eGTM().fit(X_train).transform(X_test)
predicted_labels = eGTC().fit(X_train, y_train).predict(X_test)
predicted_values = eGTR().fit(X_train, y_train).predict(X_test)
```

## Visualisation

ugtm no longer bundles a plotting module. The outputs (`matMeans`, `matModes`, `matR`) are plain NumPy arrays — use matplotlib directly:

```python
import matplotlib.pyplot as plt

gtm = ugtm.runGTM(data=data)
coords = gtm.matMeans  # shape (n_samples, 2)

plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Spectral_r")
plt.colorbar()
plt.title("GTM map")
plt.show()
```

For richer examples (landscapes, class maps, projections) see the [readthedocs documentation](https://ugtm.readthedocs.io/).

## Citing ugtm

Cite ugtm and the [following paper](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.235/):

```bibtex
@ARTICLE{Gaspar2018-qt,
  title   = "ugtm: A Python Package for Data Modeling and Visualization Using
             Generative Topographic Mapping",
  author  = "Gaspar, H{\'e}l{\'e}na Alexandra",
  journal = "Journal of Open Research Software",
  volume  =  6,
  pages   = "215",
  month   =  dec,
  year    =  2018
}
```

## Author

Héléna A. Gaspar — hagax8@gmail.com — [github.com/hagax8](https://github.com/hagax8)
