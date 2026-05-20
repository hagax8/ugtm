ugtm: Generative Topographic Mapping with Python
=================================================

GTM (Generative Topographic Mapping) is a dimensionality reduction algorithm
(like t-SNE, LLE, etc.) created by Bishop et al. and a probabilistic
counterpart of Kohonen maps. ugtm implements GTM and GTM-based prediction
algorithms, including a kernel variant (kGTM), classification (GTC) and
regression (GTR) maps, sklearn-compatible estimators, and repeated
cross-validation.

Full documentation: https://ugtm.readthedocs.io/


Quick start
-----------

::

    import ugtm
    import numpy as np

    data   = np.random.randn(100, 50)
    labels = np.random.choice([1, 2], size=100)

    gtm = ugtm.runGTM(data=data)

    coordinates = gtm.matMeans   # mean positions  (n_samples, 2)
    modes       = gtm.matModes   # mode positions  (n_samples, 2)
    resp        = gtm.matR       # responsibilities (n_samples, n_nodes)


sklearn-compatible estimators
------------------------------

::

    from ugtm import eGTM, eGTC, eGTR

    transformed      = eGTM().fit(X_train).transform(X_test)
    predicted_labels = eGTC().fit(X_train, y_train).predict(X_test)
    predicted_values = eGTR().fit(X_train, y_train).predict(X_test)


Large datasets: incremental GTM (iGTM)
---------------------------------------

For datasets too large to hold the full N×K responsibility matrix in RAM,
use iGTM (Gaspar et al. 2014). Data is processed in blocks; only two small
accumulators are kept per iteration instead of the full N×K matrix::

    from ugtm import runIGTM, eIGTM

    # Low-level wrapper — same interface as runGTM
    model = runIGTM(data, n_blocks=10)
    coordinates = model.matMeans   # (n_samples, 2)

    # sklearn transformer — n_blocks=0 chooses block size automatically
    transformed = eIGTM().fit(X_train).transform(X_test)

    # Block-wise projection for large test sets (generator, bounded memory)
    for block in eIGTM().fit(X_train).transform_blocks(X_test, block_size=1000):
        pass  # process each (block_size, 2) chunk here


Visualisation
-------------

ugtm outputs are plain NumPy arrays — use any plotting library::

    import matplotlib.pyplot as plt

    gtm    = ugtm.runGTM(data=data)
    coords = gtm.matMeans

    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Spectral_r")
    plt.colorbar()
    plt.show()

See https://ugtm.readthedocs.io/ for richer examples.


Predictions and cross-validation
---------------------------------

::

    # GTM classification / regression
    predicted = ugtm.GTC(train=train, test=test, labels=labels)
    predicted = ugtm.GTR(train=train, test=test, labels=activity)

    # Repeated cross-validation
    ugtm.crossvalidateGTC(data=train, labels=labels, s=1, regul=1)
    ugtm.crossvalidateGTR(data=train, labels=activity, s=1, regul=1)


References
----------

1. GTM algorithm — Bishop et al. (1998)
2. Kernel GTM — https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2010-44.pdf
3. GTM classification models — https://www.ncbi.nlm.nih.gov/pubmed/24320683
4. GTM regression models — https://www.ncbi.nlm.nih.gov/pubmed/27490381
5. ugtm paper — https://openresearchsoftware.metajnl.com/articles/10.5334/jors.235/
6. Incremental GTM — Gaspar et al. (2014), Chemical Data Visualization and Analysis with Incremental GTM: Big Data Challenge
