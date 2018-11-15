"""Functions to run GTM models.
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from . import ugtm_preprocess
from . import ugtm_classes
from . import ugtm_core


def initialize(data, k, m, s, random_state=1234):
    r"""Initializes a GTM model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    k : int
        Sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int
        Sqrt of the number of RBF centers.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
    s : float
        RBF width factor.
        One of four GTM hyperparameters (k, m, s, regul).
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    random_state : int, optional (default = 1234)
        Random state.

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.InitialGTM`
        Initial GTM model (not optimized).

    Notes
    =====
    We use approximately the same notations as in the original GTM paper
    by C. Bishop et al.
    The initialization process is the following:

        1. GTM grid parameters:
           number of nodes = k*k, number of rbf centers = m*m
        2. Create node matrix X
           (matX, meshgrid of shape (k*k,2))
        3. Create rbf centers matrix M
           (matM, meshgrid of shape (m*m, 2))
        4. Initialize rbf width
           (rbfWidth, :func:`~ugtm.ugtm_core.computeWidth`)
        5. Create rbf matrix :math:`\Phi`
           (matPhiMPlusOne, :func:`~ugtm.ugtm_core.createPhiMatrix`)
        6. Perform PCA on the data using sklearn's PCA function
        7. Set U matrix to 2 first principal axes of data cov. matrix (matU)
        8. Initialize parameter matrix W using U and :math:`\Phi`
           (matW, :func:`~ugtm.ugtm_core.createWMatrix`)
        9. Initialize manifold Y using W and :math:`\Phi`
           (matY, :func:`~ugtm.ugtm_core.createYMatrixInit`)
        10. Set noise variance parameter (betaInv, :func:`~ugtm.ugtm_core.evalBetaInv`)
            to the largest between:
            (1) the 3rd eigenvalue of the data covariance matrix
            (2) half the average distance between centers of Gaussian components.
        11. Store initial GTM model in InitialGTM object

    """
    n_dimensions = data.shape[1]
    n_nodes = k*k
    n_rbf_centers = m*m
    x = np.linspace(-1, 1, k)
    matX = np.transpose(np.meshgrid(x, x)).reshape(k*k, 2)
    x = np.linspace(-1, 1, m)
    matM = np.transpose(np.meshgrid(x, x)).reshape(m*m, 2)
    if m == 1:
        matM = np.array([[0.0, 0.0]])
    if k == 1:
        matX = np.array([[0.0, 0.0]])
    rbfWidth = ugtm_core.computeWidth(matM, n_rbf_centers, s)
    matPhiMPlusOne = ugtm_core.createPhiMatrix(
        matX, matM, n_nodes, n_rbf_centers, rbfWidth)
    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(data)
    matU = (pca.components_.T * np.sqrt(pca.explained_variance_))[:, 0:2]
    betaInv = pca.explained_variance_[2]
    Uobj = ugtm_classes.ReturnU(matU, betaInv)
    matW = ugtm_core.createWMatrix(matX, matPhiMPlusOne, Uobj.matU,
                                   n_dimensions, n_rbf_centers)
    matY = ugtm_core.createYMatrixInit(data, matW, matPhiMPlusOne)
    betaInv = Uobj.betaInv
    betaInv = ugtm_core.evalBetaInv(matY, Uobj.betaInv,
                                    random_state=random_state)
    return ugtm_classes.InitialGTM(matX, matM, n_nodes,
                                   n_rbf_centers, rbfWidth,
                                   matPhiMPlusOne,
                                   matW, matY, betaInv, n_dimensions)


def optimize(data, initialModel, regul, niter, verbose=True):
    r"""Optimizes a GTM model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    initialModel : instance of :class:`~ugtm.ugtm_classes.InitialGTM`
        PCA-initialized GTM model.
        The initial model is separated from the optimized model
        so that different data sets can be potentially used for initialization
        and optimization.
    regul : float
        Regularization coefficient.
    niter : int
        Number of iterations for EM algorithm.
    verbose : bool, optional (default = True)
        Verbose mode (outputs loglikelihood values during EM algorithm).

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized GTM model.

    Notes
    =====
    We use approximately the same notations as in the original GTM paper
    by C. Bishop et al.
    The GTM optimization process is the following:

        1. Create distance matrix D between manifold and data matrix
           (matD, :func:`~ugtm.ugtm_core.createDistanceMatrix`)

        2. Until convergence (:math:`\Delta(log likelihood) \leq 0.0001`):

            1. Update data distribution P
               (matP, :func:`~ugtm.ugtm_core.createPMatrix`)
            2. Update responsibilities R
               (matR, :func:`~ugtm.ugtm_core.createRMatrix`)
            #. Update diagonal matrix G
               (matG, :func:`~ugtm.ugtm_core.createGMatrix`)
            #. Update parameter matrix W
               (matW, :func:`~ugtm.ugtm_core.optimWMatrix`)
            #. Update manifold matrix Y
               (matY, :func:`~ugtm.ugtm_core.createYMatrix`)
            #. Update distance matrix D
               (matD, :func:`~ugtm.ugtm_core.createDistanceMatrix`)
            #. Update noise variance parameter :math:`\beta^{-1}`
               (betaInv, :func:`~ugtm.ugtm_core.optimBetaInv`)
            #. Estimate log likelihood and check for convergence
               (:func:`~ugtm.ugtm_core.computelogLikelihood`)

        3. Compute 2D GTM representation 1: means
           (matMeans, :func:`~ugtm.ugtm_core.meanPoint`)

        4. Compute 2D GTM representation 2: modes
           (matModes, :func:`~ugtm.ugtm_core.modePoint`)

        5. Store GTM model in OptimizedGTM object
    """
    matD = ugtm_core.createDistanceMatrix(initialModel.matY, data)
    matY = initialModel.matY
    betaInv = initialModel.betaInv
    i = 1
    diff = 1000
    converged = 0
    while i < (niter+1) and (converged < 4):
        # expectation
        matP = ugtm_core.createPMatrix(matD, betaInv,
                                       initialModel.n_dimensions)
        matR = ugtm_core.createRMatrix(matP)
        # maximization
        matG = ugtm_core.createGMatrix(matR)
        matW = ugtm_core.optimWMatrix(
            matR, initialModel.matPhiMPlusOne, matG, data, betaInv, regul)
        matY = ugtm_core.createYMatrix(matW, initialModel.matPhiMPlusOne)
        matD = ugtm_core.createDistanceMatrix(matY, data)
        betaInv = ugtm_core.optimBetaInv(
            matR, matD, initialModel.n_dimensions)
        # objective function
        if i == 1:
            loglike = ugtm_core.computelogLikelihood(
                matP, betaInv, initialModel.n_dimensions)
        else:
            loglikebefore = loglike
            loglike = ugtm_core.computelogLikelihood(
                matP, betaInv, initialModel.n_dimensions)
            diff = abs(loglikebefore-loglike)
        if diff <= 0.0001:
            converged += 1
        else:
            converged = 0
        if verbose:
            print("Iter ", i, " Err: ", loglike)
        i += 1
    # final iteration to make sure matR fits matD
    if verbose == 1:
        if converged >= 3:
            print("Converged: ", loglike)
    if converged >= 3:
        has_converged = True
    else:
        has_converged = False
    matP = ugtm_core.createPMatrix(matD, betaInv, initialModel.n_dimensions)
    matR = ugtm_core.createRMatrix(matP)
    matMeans = ugtm_core.meanPoint(matR, initialModel.matX)
    matModes = ugtm_core.modePoint(matR, initialModel.matX)
    return ugtm_classes.OptimizedGTM(matW, matY, matP.T, matR.T,
                                     betaInv, matMeans,
                                     matModes,
                                     initialModel.matX,
                                     initialModel.n_dimensions,
                                     has_converged)


def projection(optimizedModel, new_data):
    r"""Project test set on optimized GTM model. No pre-processing involved.

    Parameters
    ==========
    optimizedModel : instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized GTM model, built using training set (train).
    new_data : array of shape (n_test, n_dimensions)
        Test data matrix.

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Returns an instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        corresponding to the projected test set.

    Notes
    =====
    The new_data must have been through exactly the same preprocessing
    as the data used to obtained the optimized GTM model. To get a function
    doing the preprocessing as well as projection on the map, cf.
    :func:`~ugtm.ugtm_gtm.transform`.
    """
    matD = ugtm_core.createDistanceMatrix(optimizedModel.matY, new_data)
    matP = ugtm_core.createPMatrix(matD, optimizedModel.betaInv,
                                   optimizedModel.n_dimensions)
    matR = ugtm_core.createRMatrix(matP)
    # loglike = ugtm_core.computelogLikelihood(
    # matP, optimizedModel.betaInv, optimizedModel.n_dimensions)
    matMeans = ugtm_core.meanPoint(matR, optimizedModel.matX)
    matModes = ugtm_core.modePoint(matR, optimizedModel.matX)
    return ugtm_classes.OptimizedGTM(optimizedModel.matW, optimizedModel.matY,
                                     matP.T, matR.T,
                                     optimizedModel.betaInv,
                                     matMeans, matModes,
                                     optimizedModel.matX,
                                     optimizedModel.n_dimensions,
                                     optimizedModel.converged)


def transform(optimizedModel, train, test, doPCA=False, n_components=-1,
              missing=True,
              missing_strategy="median", random_state=1234, process=True):
    r""" Preprocess and project test set on optimized GTM model.

    Parameters
    ==========
    optimizedModel : instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized GTM model, built using training set (train).
    train : array of shape (n_train, n_dimensions)
        Training data matrix.
    test : array of shape (n_test, n_dimensions)
        Test data matrix.
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    n_components : int, optional (default = -1)
        Number of components for PCA pre-processing.
        If set to -1, keep principal components
        accounting for 80% of data variance.
    missing : bool, optional (default = True)
        Replace missing values (calls scikit-learn functions).
    missing_strategy : str (default = 'median')
        Scikit-learn missing data strategy.
    random_state : int (default = 1234)
        Random state.
    process : bool (default = True)
        Apply preprocessing (missing, PCA) to train set and
        use values from train set to preprocess test set.

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Returns an instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        corresponding to the projected test set.
    """
    if process is True:
        processed = ugtm_preprocess.processTrainTest(train, test, doPCA=doPCA,
                                                     n_components=n_components,
                                                     missing=missing,
                                                     missing_strategy=missing_strategy,
                                                     random_state=random_state)
    if process is True:
        return projection(optimizedModel, processed.test)
    else:
        return projection(optimizedModel, test)


def runGTM(data, k=16, m=4, s=0.3, regul=0.1,
           doPCA=False, n_components=-1,
           missing=True, missing_strategy="median",
           random_state=1234,
           niter=200, verbose=False):
    r"""Run GTM (wrapper for initialize + optimize).

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        m is the qrt of the number of RBF centers.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
    s : float, optional (default = 0.3)
        RBF width factor.
        One of four GTM hyperparameters (k, m, s, regul).
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = 0.1)
        One of four GTM hyperparameters (k, m, s, regul).
        Regularization coefficient.
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    n_components : int, optional (default = -1)
        Number of components for PCA pre-processing.
        If set to -1, keep principal components
        accounting for 80% of data variance.
    missing : bool, optional (default = True)
        Replace missing values (calls scikit-learn functions).
    missing_strategy : str (default = 'median')
        Scikit-learn missing data strategy.
    random_state : int (default = 1234)
        Random state.
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    verbose : bool, optional (default = False)
        Verbose mode (outputs loglikelihood values during EM algorithm).

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized GTM model.

    Notes
    =====
    We use approximately the same notations as in the original GTM paper
    by C. Bishop et al.
    The initialization + optimization process the following:

        1. GTM initialization:

            1. GTM grid parameters:
               number of nodes = k*k, number of rbf centers = m*m
            2. Create node matrix X
               (matX, meshgrid of shape (k*k,2))
            3. Create rbf centers matrix M
               (matM, meshgrid of shape (m*m, 2))
            4. Initialize rbf width
               (rbfWidth, :func:`~ugtm.ugtm_core.computeWidth`)
            5. Create rbf matrix :math:`\Phi`
               (matPhiMPlusOne, :func:`~ugtm.ugtm_core.createPhiMatrix`)
            6. Perform PCA on the data using sklearn's PCA function
            7. Set U matrix to 2 first principal axes of data cov. matrix (matU)
            8. Initialize parameter matrix W using U and :math:`\Phi`
               (matW, :func:`~ugtm.ugtm_core.createWMatrix`)
            9. Initialize manifold Y using W and :math:`\Phi`
               (matY, :func:`~ugtm.ugtm_core.createYMatrixInit`)
            10. Set noise variance parameter (betaInv, :func:`~ugtm.ugtm_core.evalBetaInv`)
                to the largest between:
                (1) the 3rd eigenvalue of the data covariance matrix
                (2) half the average distance between centers of Gaussian components.

        2. GTM optimization:

            1. Create distance matrix D between manifold and data matrix
               (matD, :func:`~ugtm.ugtm_core.createDistanceMatrix`)
            2. Until convergence (:math:`\Delta(log likelihood) \leq 0.0001`):

                1. Update data distribution P
                   (matP, :func:`~ugtm.ugtm_core.createPMatrix`)
                2. Update responsibilities R
                   (matR, :func:`~ugtm.ugtm_core.createRMatrix`)
                #. Update diagonal matrix G
                   (matG, :func:`~ugtm.ugtm_core.createGMatrix`)
                #. Update parameter matrix W
                   (matW, :func:`~ugtm.ugtm_core.optimWMatrix`)
                #. Update manifold matrix Y
                   (matY, :func:`~ugtm.ugtm_core.createYMatrix`)
                #. Update distance matrix D
                   (matD, :func:`~ugtm.ugtm_core.createDistanceMatrix`)
                #. Update noise variance parameter :math:`\beta^{-1}`
                   (betaInv, :func:`~ugtm.ugtm_core.optimBetaInv`)
                #. Estimate log likelihood and check for convergence
                   (:func:`~ugtm.ugtm_core.computelogLikelihood`)

            3. Compute 2D GTM representation 1: means
               (matMeans, :func:`~ugtm.ugtm_core.meanPoint`)
            4. Compute 2D GTM representation 2: modes
               (matModes, :func:`~ugtm.ugtm_core.modePoint`)
            5. Store GTM model in OptimizedGTM object
    """

    if k == 0:
        k = int(np.sqrt(5*np.sqrt(data.shape[0])))+2
    if m == 0:
        m = int(np.sqrt(k))

    data = ugtm_preprocess.pcaPreprocess(data=data, doPCA=doPCA,
                                         n_components=n_components,
                                         missing=missing,
                                         missing_strategy=missing_strategy,
                                         random_state=random_state)
    initialModel = initialize(data, k, m, s, random_state)
    optimizedModel = optimize(
        data, initialModel, regul, niter, verbose=verbose)
    return optimizedModel
