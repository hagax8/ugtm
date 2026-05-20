"""Functions to initialize and optimize incremental GTM (iGTM) models.

Reference: Gaspar et al. 2014, "Chemical Data Visualization and Analysis
with Incremental GTM: Big Data Challenge."
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import math
import numpy as np
from . import ugtm_core
from . import ugtm_classes
from . import ugtm_gtm
from . import ugtm_preprocess


def _auto_n_blocks(n_individuals, block_size=5000):
    """Return ceil(N / block_size), minimum 1."""
    return max(1, math.ceil(n_individuals / block_size))


def _make_block_indices(n_individuals, n_blocks):
    """Return list of (start, end) slice pairs covering all N individuals."""
    block_size = math.ceil(n_individuals / n_blocks)
    indices = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n_individuals)
        if start < n_individuals:
            indices.append((start, end))
    return indices


def optimize_igtm(data, initialModel, regul, niter, n_blocks, verbose=False):
    r"""Optimizes an iGTM model using single-pass block-wise EM.

    Each EM iteration makes one pass over the data in ``n_blocks`` blocks.
    The sufficient statistics for the M-step
    (:math:`\mathbf{g}` and :math:`\mathbf{R}^T \mathbf{X}`) and for the
    :math:`\beta^{-1}` update are accumulated block-by-block so that the
    full N×K responsibility matrix is never held in memory simultaneously.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix (already preprocessed).
    initialModel : instance of :class:`~ugtm.ugtm_classes.InitialGTM`
        PCA-initialized GTM model.
    regul : float
        Regularization coefficient.
    niter : int
        Maximum number of EM iterations.
    n_blocks : int
        Number of data blocks. Use 1 for standard single-block behaviour.
    verbose : bool, optional (default = False)
        Print log-likelihood at each iteration.

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized iGTM model. ``matR`` and ``matP`` are ``None`` (not
        retained to save memory; call :func:`~ugtm.ugtm_gtm.projection`
        to recompute them for any data subset).
    """
    N = data.shape[0]
    blocks = _make_block_indices(N, n_blocks)
    Phi = initialModel.matPhiMPlusOne          # (n_nodes, n_rbf+1)
    n_dims = initialModel.n_dimensions
    n_nodes = initialModel.n_nodes
    matX = initialModel.matX                   # (n_nodes, 2)
    prior = 1.0 / n_nodes
    placeholder = 50

    matY = initialModel.matY
    betaInv = initialModel.betaInv

    i = 1
    diff = 1000.0
    converged = 0
    loglike = 0.0

    while i <= niter and converged < 4:
        g_vec = np.zeros(n_nodes, dtype=np.float64)
        RT_acc = np.zeros((n_nodes, n_dims), dtype=np.float64)
        weighted_D_sum = 0.0
        loglike_num = np.longdouble(0.0)

        # constant factor for log-likelihood (same across blocks per iteration)
        constante = np.longdouble(np.power(
            (1.0 / betaInv) / (2.0 * np.pi),
            np.minimum(n_dims / 2.0, placeholder)
        ))

        # E-step: one pass over blocks, accumulate sufficient statistics
        for start, end in blocks:
            X_b = data[start:end]
            D_b = ugtm_core.createDistanceMatrix(matY, X_b)
            P_b = ugtm_core.createPMatrix(D_b, betaInv, n_dims)
            R_b = ugtm_core.createRMatrix(P_b)

            g_vec += R_b.sum(axis=1)
            RT_acc += R_b @ X_b
            weighted_D_sum += float(np.sum(R_b * D_b))
            loglike_num += np.sum(np.log(np.maximum(
                np.sum(constante * P_b, axis=0) * prior,
                np.finfo(np.longdouble).tiny
            )))

        new_loglike = float(-loglike_num / N)
        if i > 1:
            diff = abs(loglike - new_loglike)
        loglike = new_loglike

        if verbose:
            print("Iter", i, " Err:", loglike)

        # M-step
        matW = ugtm_core.optimWMatrixAcc(g_vec, RT_acc, Phi, betaInv, regul)
        matY = ugtm_core.createYMatrix(matW, Phi)
        betaInv = weighted_D_sum / (N * n_dims)

        if diff <= 0.0001:
            converged += 1
        else:
            converged = 0
        i += 1

    has_converged = converged >= 3
    if verbose and has_converged:
        print("Converged:", loglike)

    # Final pass: compute matMeans and matModes block-by-block with final W
    matMeans = np.zeros((N, 2), dtype=np.float64)
    matModes = np.zeros((N, 2), dtype=np.float64)
    for start, end in blocks:
        X_b = data[start:end]
        D_b = ugtm_core.createDistanceMatrix(matY, X_b)
        P_b = ugtm_core.createPMatrix(D_b, betaInv, n_dims)
        R_b = ugtm_core.createRMatrix(P_b)
        matMeans[start:end] = R_b.T @ matX
        matModes[start:end] = matX[np.argmax(R_b, axis=0), :]

    return ugtm_classes.OptimizedGTM(
        matW, matY,
        None,   # matP: not retained (N×n_nodes)
        None,   # matR: not retained (N×n_nodes)
        betaInv, matMeans, matModes, matX, n_dims, has_converged
    )


def runIGTM(data, k=16, m=4, s=0.3, regul=0.1, n_blocks=0,
            doPCA=False, n_components=-1,
            missing=True, missing_strategy="median",
            random_state=1234, niter=200, verbose=False):
    r"""Run iGTM algorithm (wrapper for initialize + optimize_igtm).

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    k : int, optional (default = 16)
        Sqrt of the number of GTM nodes.
        Set to 0 to auto-compute as ``sqrt(5*sqrt(N))+2``.
    m : int, optional (default = 4)
        Sqrt of the number of RBF centers.
        Set to 0 to auto-compute as ``sqrt(k)``.
    s : float, optional (default = 0.3)
        RBF width factor.
    regul : float, optional (default = 0.1)
        Regularization coefficient.
    n_blocks : int, optional (default = 0)
        Number of data blocks. 0 = auto (``ceil(N / 5000)``).
    doPCA : bool, optional (default = False)
        Apply PCA preprocessing.
    n_components : int, optional (default = -1)
        Number of PCA components (−1 = keep 80% variance).
    missing : bool, optional (default = True)
        Impute missing values.
    missing_strategy : str (default = 'median')
        Imputation strategy passed to sklearn's SimpleImputer.
    random_state : int (default = 1234)
        Random state.
    niter : int, optional (default = 200)
        Maximum EM iterations.
    verbose : bool, optional (default = False)
        Print log-likelihood at each iteration.

    Returns
    =======
    instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized iGTM model.
    """
    if k == 0:
        k = int(np.sqrt(5 * np.sqrt(data.shape[0]))) + 2
    if m == 0:
        m = int(np.sqrt(k))
    if n_blocks == 0:
        n_blocks = _auto_n_blocks(data.shape[0])

    data = ugtm_preprocess.pcaPreprocess(
        data=data, doPCA=doPCA, n_components=n_components,
        missing=missing, missing_strategy=missing_strategy,
        random_state=random_state
    )

    initialModel = ugtm_gtm.initialize(data, k, m, s, random_state)
    return optimize_igtm(data, initialModel, regul, niter, n_blocks, verbose)
