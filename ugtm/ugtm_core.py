"""Core linear algebra operations for GTM and kGTM
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale


def createYMatrixInit(data, matW, matPhiMPlusOne):
    r"""Creates initial manifold matrix (Y).

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    matW : array of shape (n_dimensions, n_rbf_centers+1)
        Parameter matrix (PCA-initialized).
    matPhiMPlusOne : array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.

    Returns
    =======
    array of shape (n_dimensions, n_nodes)
        Manifold in n-dimensional space (projection of matX in data space);
        A point matY[:,i] is a center of Gaussian component in data space.
        :math:`\mathbf{Y}=\mathbf{W}\mathbf{\Phi}^T`
    """
    shap1 = matW.shape[0]
    shap2 = matPhiMPlusOne.shape[0]
    TheMeans = data.mean(0)
    DMmeanMatrix = np.zeros([shap1, shap2])
    for i in range(shap1):
        for j in range(shap2):
            DMmeanMatrix[i, j] = TheMeans[i]
    MatY = np.dot(matW, np.transpose(matPhiMPlusOne))
    MatY = MatY + DMmeanMatrix
    return(MatY)


def createPhiMatrix(matX, matM, numX, numM, sigma):
    r"""Creates matrix of RBF functions.

    Parameters
    ==========
    matX : array of shape (n_nodes, 2)
        Coordinates of nodes defining a grid in the 2D space.
    matM : array of shape (n_rbf_centers, 2)
        Coordinates of radial basis function (RBF) centers,
        defining a grid in the 2D space.
    numX : int
        Number of nodes (n_nodes).
    numM : int
        Number of RBF centers (n_rbf_centers)
    sigma : float
        RBF width factor.

    Returns
    =======
    array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.
    """
    Result = np.zeros([numX, numM + 1])
    for i in range(numX):
        for j in range(numM):
            Coo1 = (matX[i][0] - matM[j][0]) * (matX[i][0] - matM[j][0])
            Coo2 = (matX[i][1] - matM[j][1]) * (matX[i][1] - matM[j][1])
            Distance = Coo1 + Coo2
            Result[i, j] = np.exp(-(Distance) / (2 * sigma))
    for i in range(numX):
        Result[i][numM] = 1
    return(Result)


def computeWidth(matM, numM, sigma):
    r"""Initializes radial basis function width using hyperparameter sigma.

    Parameters
    ==========
    matM : array of shape (n_rbf_centers, 2)
        Coordinates of radial basis function (RBF) centers,
        defining a grid in the 2D space.
    numM : int
        Number of RBF centers (n_rbf_centers)
    sigma : float
        RBF width factor.

    Returns
    =======
    float
        Initial radial basis function (RBF) width.
    """
    Result = 0.0
    if numM <= 1:
        return(sigma)
    else:
        Distances = np.zeros([numM, numM])
        mins = np.zeros([numM, 1])
        maxs = np.zeros([numM, 1])
        for i in range(numM):
            for j in range(numM):
                Coo1 = (matM[i][0] - matM[j][0]) * (matM[i][0] - matM[j][0])
                Coo2 = (matM[i][1] - matM[j][1]) * (matM[i][1] - matM[j][1])
                Distances[i, j] = Coo1 + Coo2
        for i in range(numM):
            mins[i] = np.min(Distances[i][np.nonzero(Distances[i])])
        for i in range(numM):
            maxs[i] = np.max(Distances[i][np.nonzero(Distances[i])])
        if (sigma > 0.0):
            Result = sigma * np.mean(mins)
        else:
            Result = np.max(maxs)
        return(Result)


def createWMatrix(matX, matPhiMPlusOne, matU, n_dimensions, n_rbf_centers):
    r"""Creates PCA-initialized parameter matrix W.

    Parameters
    ==========
    matX : array of shape (n_nodes, 2)
        Coordinates of nodes defining a grid in the 2D space.
    matPhiMPlusOne: array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.
    matU : array of shape (n_dimensions, 2)
        2 first principal axes of data covariance matrix.
    n_dimensions: int
        Data space dimensionality (number of variables).
    n_rbf_centers : int
        Number of RBF centers.
    sigma : float
        RBF width factor.

    Returns
    =======
    array of shape (n_dimensions, n_rbf_centers+1)
        Parameter matrix W (PCA-initialized).
    """

    NormX = scale(matX, axis=0, with_mean=True, with_std=True)
    myProd = np.dot(matU, np.transpose(NormX))
    tinv = np.linalg.solve(matPhiMPlusOne.T.dot(
        matPhiMPlusOne), matPhiMPlusOne.T)
    Result = np.dot(myProd, np.transpose(tinv))
    return(Result)


def createYMatrix(matW, matPhiMPlusOne):
    r"""Updates manifold matrix (Y) using new parameter matrix (W).

    Parameters
    ==========
    matW : array of shape (n_dimensions, n_rbf_centers+1)
        Parameter matrix (PCA-initialized).
    matPhiMPlusOne : array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.

    Returns
    =======
    array of shape (n_dimensions, n_nodes)
        Manifold in n-dimensional space (projection of matX in data space);
        A point matY[:,i] is a center of Gaussian component in data space.
        :math:`\mathbf{Y}=\mathbf{W}\mathbf{\Phi}^T`
    """
    Result = np.dot(matW, np.transpose(matPhiMPlusOne))
    return(Result)


def createDistanceMatrix(matY, data):
    r"""Computes distances between manifold centers and data vectors.

    Parameters
    ==========
    matY : array of shape (n_dimensions, n_nodes)
        Manifold in n-dimensional space (projection of matX in data space);
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.

    Returns
    =======
    array of shape (n_nodes, n_individuals)
        Matrix of squared Euclidean distances between manifold and data.
    """
    Result = distance.cdist(matY.T, data, metric='sqeuclidean')
    return(Result)


def KERNELcreateDistanceMatrix(data, matL, matPhiMPlusOne):
    r"""Computes distances between data and manifold for kernel algorithm.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    matL : array of shape (n_individuals, n_rbf_centers+1)
        Parameter matrix (regul).
    matPhiMPlusOne: array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.

    Returns
    =======
    array of shape (n_nodes, n_individuals)
        Matrix of distances between manifold and data.
    """
    n_nodes = matPhiMPlusOne.shape[0]
    n_individuals = data.shape[0]
    Result = np.zeros([n_nodes, n_individuals])
    thefloat = 0.0
    for i in range(n_nodes):
        LPhim = np.dot(matL, matPhiMPlusOne[i])
        thefloat = np.dot(np.dot(LPhim, data), LPhim)
        for j in range(n_individuals):
            Result[i, j] = data[j, j] + thefloat - 2*(np.dot(data[j], LPhim))
    return(Result)


def exp_normalize(x):
    r"""Exp-normalize trick: compute exp(x-max(x))

    Parameters
    ==========
        2D array
            An array x

    Returns
    =======
        2D array
            y = exp(x-max(x))
    """
    y = np.array([], dtype=np.longdouble)
    y = x - np.expand_dims(np.max(x, axis=0), 0)
    y = np.exp(y)
    return(y)


def createPMatrix(matD, betaInv, n_dimensions):
    r"""Computes data distribution matrix = exp(-(parameter)*distances).

    Parameters
    ==========
    matD : array of shape (n_nodes, n_individuals)
        Matrix of squared Euclidean distances between manifold and data.
    betaInv : float
        Noise variance parameter for the data distribution.
        Written as :math:`\beta^{-1}` in the original paper.
    n_dimensions : int
            Data space dimensionality (number of variables).

    Returns
    =======
    array of shape (n_nodes, n_individuals)
        Data distribution with variance betaInv (transformed: exp(x-max(x)))

    Notes
    =====
    Important: this data distribution is not exact per se
    and is to be used as input
    for createRMatrix (responsibilities).
    """
    matP = np.array([], dtype=np.longdouble)
    beta = 1/betaInv
    # exp_normalize computes exp(x-max(x)) to avoid overflow for createRMatrix
    matP = exp_normalize(-(beta/2)*matD)
    return(matP)


def createRMatrix(matP):
    r"""Computes responsibilities (posterior probabilities).

    Parameters
    ==========
    matP : array of shape (n_nodes, n_individuals)
        Data distribution with variance betaInv (transformed: exp(x-max(x)))

    Returns
    =======
    array of shape (n_nodes, n_individuals)
        Posterior probabilities (responsibilities).
    """
    matR = np.array([], dtype=np.longdouble)
    sums = np.sum(matP, axis=0)
    matR = (matP) / (sums[None, :])
    return(matR)


def createGMatrix(matR):
    r"""Creates the G diagonal matrix from responsibilities (R)

    Parameters
    ==========
    matR : array of shape (n_nodes, n_individuals)
        Posterior probabilities (responsibilities).

    Returns
    =======
    array of shape (n_nodes, n_nodes)
        Diagonal matrix with elements
        :math:`G_{ii}=\sum_{n}^{n\_individuals} R_{in}`.
    """
    sums = np.sum(matR, axis=1)
    matG = np.diag(sums)
    return(matG)


def optimWMatrix(matR, matPhiMPlusOne, matG, data, betaInv, regul):
    r"""Updates parameter matrix W.

    Parameters
    ==========
    matR : array of shape (n_nodes, n_individuals)
        Posterior probabilities (responsibilities).
    matPhiMPlusOne: array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.
    matG : array of shape (n_nodes, n_nodes)
        Diagonal matrix with elements
        :math:`G_{ii}=\sum_{n}^{n\_individuals} R_{in}`.
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    betaInv : float
        Noise variance parameter for the data distribution.
        Written as :math:`\beta^{-1}` in the original paper.
    regul : float
        Regularization coefficient.

    Returns
    =======
    array of shape (n_dimensions, n_rbf_centers+1)
        Updated parameter matrix W.
    """
    n_rbf_centersP = matPhiMPlusOne.shape[1]
    LBmat = np.zeros([n_rbf_centersP, n_rbf_centersP])
    PhiGPhi = np.dot(
        np.dot(np.transpose(matPhiMPlusOne), matG), matPhiMPlusOne)
    for i in range(n_rbf_centersP):
        LBmat[i][i] = regul * betaInv
    PhiGPhiLB = PhiGPhi + LBmat
    Ginv = np.linalg.inv(PhiGPhiLB)
    matW = np.transpose(
        np.dot(np.dot(np.dot(Ginv, np.transpose(matPhiMPlusOne)), matR), data))
    return(matW)


def optimLMatrix(matR, matPhiMPlusOne, matG, betaInv, regul):
    r"""Updates parameter matrix regul for kernel GTM.

    Parameters
    ==========
    matR : array of shape (n_nodes, n_individuals)
        Posterior probabilities (responsibilities).
    matPhiMPlusOne: array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.
    matG : array of shape (n_nodes, n_nodes)
        Diagonal matrix with elements
        :math:`G_{ii}=\sum_{n}^{n\_individuals} R_{in}`.
    betaInv : float
        Noise variance parameter for the data distribution.
        Written as :math:`\beta^{-1}` in the original paper.
    regul : float
        Regularization coefficient.

    Returns
    =======
    array of shape (n_individuals, n_rbf_centers+1)
        Updated parameter matrix regul.
    """
    n_rbf_centersP = matPhiMPlusOne.shape[1]
    LBmat = np.zeros([n_rbf_centersP, n_rbf_centersP])
    PhiGPhi = np.dot(
        np.dot(np.transpose(matPhiMPlusOne), matG), matPhiMPlusOne)
    for i in range(n_rbf_centersP):
        LBmat[i][i] = regul * betaInv
    PhiGPhiLB = PhiGPhi + LBmat
    Ginv = np.linalg.inv(PhiGPhiLB)
    matW = np.transpose(
        np.dot(np.dot(Ginv, np.transpose(matPhiMPlusOne)), matR))
    return(matW)


def optimBetaInv(matR, matD, n_dimensions):
    r"""Updates noise variance parameter.

    Parameters
    ==========
    matR : array of shape (n_nodes, n_individuals)
        Posterior probabilities (responsibilities).
    matD : array of shape (n_nodes, n_individuals)
        Matrix of squared Euclidean distances between manifold and data.

    Returns
    =======
    float
        Updated noise variance parameter (:math:`\beta^{-1}`).
    """
    n_individuals = matR.shape[1]
    betaInv = np.sum(np.multiply(matR, matD))/(n_individuals*n_dimensions)
    return(betaInv)


def meanPoint(matR, matX):
    r"""Computes mean positions for data points (usual GTM output).

    Parameters
    ==========
    matR : array of shape (n_nodes, n_individuals)
        Posterior probabilities (responsibilities).
    matX : array of shape (n_nodes, 2)
        Coordinates of nodes defining a grid in the 2D space.

    Returns
    =======
    array of shape (n_individuals, 2)
        Data representation in 2D space: mean positions (usual GTM output).
    """
    matMeans = np.dot(np.transpose(matR), matX)
    return(matMeans)


def modePoint(matR, matX):
    r"""Computes modes (nodes with maximum responsibility for each data point).

    Parameters
    ==========
    matR : array of shape (n_nodes, n_individuals)
        Posterior probabilities (responsibilities).
    matX : array of shape (n_nodes, 2)
        Coordinates of nodes defining a grid in the 2D space.

    Returns
    =======
    array of shape (n_individuals, 2)
        Data representation in 2D space: modes (nodes with max responsibility).
    """
    matModes = matX[np.argmax(matR, axis=0), :]
    return(matModes)


def computelogLikelihood(matP, betaInv, n_dimensions):
    r"""Computes log likelihood = GTM objective function

    Parameters
    ==========
    matP : array of shape (n_nodes, n_individuals)
        Data distribution with variance betaInv (transformed: exp(x-max(x)))
    betaInv : float
        Noise variance parameter for the data distribution.
        Written as :math:`\beta^{-1}` in the original paper.
    n_dimensions: int
        Data space dimensionality (number of variables).

    Returns
    =======
    float
        Log likelihood.
    """
    LogLikelihood = np.longdouble()
    n_nodes = matP.shape[0]
    n_individuals = matP.shape[1]
    LogLikelihood = 0.0
    prior = 1.0/n_nodes
    placeholder = 50
    # this constant was not introduced in matP calculation
    # and is re-introduced here
    constante = np.longdouble(np.power(((1/betaInv)/(2*np.pi)),
                                       np.minimum(
        n_dimensions/2, placeholder)))
    LogLikelihood = np.sum(np.log(
        np.maximum(np.sum(constante*matP, axis=0)*prior,
                   np.finfo(np.longdouble).tiny)))
    LogLikelihood /= n_individuals
    return(-LogLikelihood)


def evalBetaInv(matY, betaInv, random_state=1234):
    r"""Decides which value to use for initial noise variance parameter.

    Parameters
    ==========
    matY : array of shape (n_dimensions, n_nodes)
        Manifold in n-dimensional space (projection of matX in data space);
    betaInv : float
        The 3rd eigenvalue of the data covariance matrix.
    random_state : int, optional
        Random state used to initialize BetaInv randomly
        in case of bad initialization.

    Returns
    =======
    float
        Noise variance parameter for the data distribution (betaInv).
        Written as :math:`\beta^{-1}` in the original paper.
        Initialized to be the larger between:
        (1) the 3rd eigenvalue of the data covariance matrix
        (function parameter),
        (2) half the average distance between centers of Gaussian components.
        In case of bad initialization (betaInv = 0),
        betaInv is set to a random value
        (a message would then be displayed on screen).
    """
    r = np.random.RandomState(random_state)
    Distances = euclidean_distances(np.transpose(matY))
    myMin = np.mean(Distances)/2
    myMin *= myMin
    if((myMin < betaInv) or (betaInv == 0)):
        betaInv = myMin
    if betaInv == 0.0:
        print("bad initialization (0 variance), "
              "setting variance to random number...")
        betaInv = abs(r.uniform(-1, 1, size=1))
    return(betaInv)


def initBetaInvRandom(matD, n_nodes, n_individuals, n_dimensions):
    r"""Computes initial noise variance parameter for kernel GTM.

    Parameters
    ==========
    matD : array of shape (n_nodes, n_individuals)
        Matrix of squared Euclidean distances between manifold and data.
    n_nodes : int
        The number of nodes defining a grid in the 2D space.
    n_individuals : int
        The number of data instances.
    n_dimensions : int
        Data space dimensionality (number of variables).

    Returns
    =======
    float
        Noise variance parameter for the data distribution (betaInv).
        Written as :math:`\beta^{-1}` in the original paper.
    """
    betaInv = np.sum(matD*1/n_nodes)/(n_individuals*n_dimensions)
    return(betaInv)
