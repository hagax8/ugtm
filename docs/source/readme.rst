tutorial
========

ugtm provides an implementation of GTM (Generative Topographic Mapping), kGTM (kernel Generative Topographic Mapping), GTM classification models (kNN, Bayes) and GTM regression models. ugtm also implements cross-validation options which can be used to compare GTM classification models to SVM classification models, and GTM regression models to SVM regression models. Typical usage::

    #!/usr/bin/env python

    import ugtm
    import numpy as np

    #generate sample data and labels: replace this with your own data
    data = np.random.randn(100, 50)
    labels = np.random.choice([1, 2], size=100)

    #build GTM map
    gtm = ugtm.runGTM(data=data, verbose=True)

    #access coordinates (means or modes)
    coordinates = gtm.matMeans
    modes = gtm.matModes


1. Import package
-----------------

Import the ugtm package, which provides GTM and kernel GTM (kGTM) maps, GTM classification models, and GTM regression models::

    import ugtm


2. Construct GTM maps (or kGTM maps)
-------------------------------------

A gtm object can be created by running the function :func:`~ugtm.ugtm_gtm.runGTM` on a dataset. Parameters for runGTM are: k = sqrt(number of nodes), m = sqrt(number of rbf centres), s = RBF width factor, regul = regularization coefficient. The number of iterations for the expectation-maximization algorithm is set to 200 by default::

    import ugtm
    import numpy as np

    train = np.random.randn(20, 10)
    test = np.random.randn(20, 10)
    labels = np.random.choice(["class1", "class2"], size=20)
    activity = np.random.randn(20, 1)

    #create a gtm object and write model
    gtm = ugtm.runGTM(train)
    gtm.write("testout1")

    #run verbose
    gtm = ugtm.runGTM(train, verbose=True)

    #to run a kernel GTM model instead:
    gtm = ugtm.runkGTM(train, doKernel=True, kernel="linear")

    #access coordinates (means or modes) and responsibilities
    gtm_coordinates = gtm.matMeans
    gtm_modes = gtm.matModes
    gtm_responsibilities = gtm.matR


3. Visualization
----------------

ugtm outputs are plain NumPy arrays (``matMeans``, ``matModes``, ``matR``), so any plotting library works. Quick example with matplotlib::

    import matplotlib.pyplot as plt

    gtm = ugtm.runGTM(data=train)
    coords = gtm.matMeans

    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Spectral_r")
    plt.colorbar()
    plt.show()

For richer interactive examples using Altair, see :doc:`visualization_examples`.


4. Incremental GTM for large datasets
--------------------------------------

For datasets too large to hold the full N×K responsibility matrix in RAM,
use :func:`~ugtm.ugtm_igtm.runIGTM` or :class:`~ugtm.ugtm_sklearn.eIGTM`.
The ``n_blocks`` parameter controls how many chunks the data is split into
(0 = auto, set to ``ceil(N / 5000)``)::

    #run incremental GTM (same interface as runGTM)
    igtm = ugtm.runIGTM(data=train, n_blocks=5, verbose=True)

    #access coordinates and responsibilities
    igtm_coordinates = igtm.matMeans
    igtm_modes       = igtm.matModes

    #sklearn-compatible transformer
    from ugtm import eIGTM
    transformed = eIGTM(n_blocks=5).fit_transform(train)

    #block-wise projection for large test sets (generator)
    model = eIGTM().fit(train)
    for block in model.transform_blocks(test, block_size=1000):
        pass  # process block here

See :doc:`eIGTM_transformer` for full details.


5. Project new data onto an existing GTM map
--------------------------------------------

New data can be projected onto an existing GTM map using the :func:`~ugtm.ugtm_gtm.transform` function. The train set is used to apply consistent preprocessing (e.g. PCA) to the test set::

    #run model on train
    gtm = ugtm.runGTM(train, doPCA=True)

    #project test data
    transformed = ugtm.transform(optimizedModel=gtm, train=train, test=test, doPCA=True)

    #access projected coordinates
    test_coordinates = transformed.matMeans
    test_modes = transformed.matModes


6. Output predictions for a test set: GTM regression (GTR) and classification (GTC)
-------------------------------------------------------------------------------------

The :func:`~ugtm.ugtm_predictions.GTR` function implements the GTM regression model and :func:`~ugtm.ugtm_predictions.GTC` function a GTM classification model::

    #continuous labels (prediction by GTM regression model)
    predicted = ugtm.GTR(train=train, test=test, labels=activity)

    #discrete labels (prediction by GTM classification model)
    predicted = ugtm.GTC(train=train, test=test, labels=labels)


7. Advanced GTM predictions with per-class probabilities
---------------------------------------------------------

Per-class probabilities for a test set can be given by the :func:`~ugtm.ugtm_predictions.advancedGTC` function::

    #get whole output model and label predictions for test set
    predicted_model = ugtm.advancedGTC(train=train, test=test, labels=labels)

    #write whole predicted model with per-class probabilities
    ugtm.printClassPredictions(predicted_model, "testout17")


8. Crossvalidation experiments
-------------------------------

Different crossvalidation experiments were implemented to compare GTC and GTR models to classical machine learning methods::

    #crossvalidation experiment: GTM classification model
    #here: set hyperparameters s=1 and regul=1 (set to -1 to optimize)
    ugtm.crossvalidateGTC(data=train, labels=labels, s=1, regul=1, n_repetitions=10, n_folds=5)

    #crossvalidation experiment: GTM regression model
    ugtm.crossvalidateGTR(data=train, labels=activity, s=1, regul=1)

    #crossvalidation experiment, k-nearest neighbours classification
    #on 2D PCA map with 7 neighbors (set to -1 to optimize)
    ugtm.crossvalidatePCAC(data=train, labels=labels, n_neighbors=7)

    #crossvalidation experiment, SVC rbf classification model (sklearn):
    ugtm.crossvalidateSVCrbf(data=train, labels=labels, C=1, gamma=1)

    #crossvalidation experiment, linear SVC classification model (sklearn):
    ugtm.crossvalidateSVC(data=train, labels=labels, C=1)

    #crossvalidation experiment, linear SVC regression model (sklearn):
    ugtm.crossvalidateSVR(data=train, labels=activity, C=1, epsilon=1)

    #crossvalidation experiment, k-nearest neighbours regression on 2D PCA map:
    ugtm.crossvalidatePCAR(data=train, labels=activity, n_neighbors=7)
