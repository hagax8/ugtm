"""Functions to initialize and optimize kernel GTM models.
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from . import ugtm_preprocess
from . import ugtm_classes
from . import ugtm_core


def initializeKernel(data, k, m, s, maxdim, random_state=1234):
    r"""Initializes a kernel GTM (kGTM) model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    k : int
        Sqrt of the number of GTM nodes.
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int
        Sqrt of the number of RBF centers.
        Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
    s : float
        RBF width factor.
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    maxdim : int
        Max boundary for internal dimensionality estimation.
    random_state : int, optional (default = 1234)
        Random state.

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.InitialGTM`
        Initial GTM model (not optimized).
    """
    n_individuals = data.shape[0]
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
    pca = PCA(random_state=random_state)
    pca.fit(data)
    matW = (pca.components_.T * np.sqrt(pca.explained_variance_)
            )[:, 0:n_rbf_centers+1]
    n_dimensions = np.searchsorted(
        pca.explained_variance_ratio_.cumsum(), 0.995)+1
    if n_dimensions > maxdim:
        n_dimensions = maxdim
    matD = ugtm_core.KERNELcreateDistanceMatrix(data, matW, matPhiMPlusOne)
    betaInv = ugtm_core.initBetaInvRandom(matD, n_nodes, n_individuals,
                                          n_dimensions)
    matY = ugtm_core.createYMatrixInit(data, matW, matPhiMPlusOne)
    return ugtm_classes.InitialGTM(matX, matM, n_nodes, n_rbf_centers,
                                   rbfWidth,
                                   matPhiMPlusOne, matW,
                                   matY, betaInv, n_dimensions)


def optimizeKernel(data, initialModel, regul, niter, verbose=True):
    r"""Optimizes a kGTM model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    initialModel : instance of :class:`~ugtm.ugtm_classes.InitialGTM`
        Initial kGTM model.
        The initial model is separate from the optimized model
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
        Optimized kGTM model.
    """
    matD = ugtm_core.KERNELcreateDistanceMatrix(
        data, initialModel.matW, initialModel.matPhiMPlusOne)
    matY = initialModel.matY
    betaInv = initialModel.betaInv
    i = 1
    diff = 1000
    converged = 0
    while (i < (niter+1)) and (converged < 4):
        # expectation
        matP = ugtm_core.createPMatrix(matD, betaInv,
                                       initialModel.n_dimensions)
        matR = ugtm_core.createRMatrix(matP)
        # maximization
        matG = ugtm_core.createGMatrix(matR)
        matW = ugtm_core.optimLMatrix(
            matR, initialModel.matPhiMPlusOne, matG, betaInv, regul)
        matY = ugtm_core.createYMatrix(matW, initialModel.matPhiMPlusOne)
        matD = ugtm_core.KERNELcreateDistanceMatrix(
            data, matW, initialModel.matPhiMPlusOne)
        betaInv = ugtm_core.optimBetaInv(matR, matD, initialModel.n_dimensions)
        # objective function
        if i == 1:
            loglike = ugtm_core.computelogLikelihood(
                matP, betaInv, initialModel.n_dimensions)
        else:
            loglikebefore = loglike
            loglike = ugtm_core.computelogLikelihood(
                matP, betaInv, initialModel.n_dimensions)
            diff = abs(loglikebefore-loglike)
        if verbose is True:
            print("Iter ", i, " Err: ", loglike)
        if diff <= 0.0001:
            converged += 1
        else:
            converged = 0
        i += 1
    if verbose is True:
        if converged >= 3:
            print("Converged: ", loglike)
    if converged >= 3:
        has_converged = True
    else:
        has_converged = False
    matY = ugtm_core.createYMatrix(matW, initialModel.matPhiMPlusOne)
    matMeans = ugtm_core.meanPoint(matR, initialModel.matX)
    matModes = ugtm_core.modePoint(matR, initialModel.matX)
    return ugtm_classes.OptimizedGTM(matW, matY, matP.T, matR.T,
                                     betaInv, matMeans,
                                     matModes,
                                     initialModel.matX,
                                     initialModel.n_dimensions,
                                     has_converged)


def runkGTM(data,  k=16, m=4, s=0.3, regul=0.1, maxdim=100,
            doPCA=False, doKernel=False, kernel="linear",
            n_components=-1,
            missing=True, missing_strategy="median",
            random_state=1234, niter=200,
            verbose=False):
    r"""Run kGTM algorithm (wrapper for initialize + optimize).

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
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = 0.1)
        One of four GTM hyperparameters (k, m, s, regul).
        Regularization coefficient.
    maxdim : int
        Max boundary for internal dimensionality estimation.
        Internal dimensionality is estimated as number of principal components
        accounting for 99.5% of data variance. If this value is higher than
        maxdim, it is replaced by maxim.
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    doKernel : bool, optional (default = False)
        If doKernel is False, the data is supposed to be a kernel already.
        If doKernel is True, a kernel will be computed from the data.
    kernel : scikit-learn kernel (default = "linear")
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
        Optimized kGTM model.
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
    if doKernel or data.shape[0] != data.shape[1]:
        data = ugtm_preprocess.chooseKernel(data, kernel)
    initialModel = initializeKernel(data, k, m, s, random_state)
    optimizedModel = optimizeKernel(
        data, initialModel, regul, niter, verbose=verbose)
    return optimizedModel
