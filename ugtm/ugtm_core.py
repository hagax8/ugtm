from __future__ import print_function
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale
# create manifold


def createYMatrixInit(data, matW, matPhiMPlusOne):
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
    Result = 0.0
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
    NormX = scale(matX, axis=0, with_mean=True, with_std=True)
    myProd = np.dot(matU, np.transpose(NormX))
    tinv = np.linalg.solve(matPhiMPlusOne.T.dot(
        matPhiMPlusOne), matPhiMPlusOne.T)
    Result = np.dot(myProd, np.transpose(tinv))
    return(Result)


def createYMatrix(matW, matPhiMPlusOne):
    Result = np.dot(matW, np.transpose(matPhiMPlusOne))
    return(Result)


def createDistanceMatrix(matY, data):
    Result = distance.cdist(matY.T, data, metric='sqeuclidean')
    return(Result)


def KERNELcreateDistanceMatrix(data, matL, matPhiMPlusOne):
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
    y = np.array([], dtype=np.longdouble)
    y = x - np.expand_dims(np.max(x, axis=0), 0)
    y = np.exp(y)
    return(y)


def createPMatrix(matD, betaInv, n_dimensions):
    matP = np.array([], dtype=np.longdouble)
    beta = 1/betaInv
    # constante = np.power(((beta)/(2*np.pi)),dim/2)
    # constante = 1
    matP = exp_normalize(-(beta/2)*matD)
    return(matP)


def createRMatrix(matP):
    matR = np.array([], dtype=np.longdouble)
    sums = np.sum(matP, axis=0)
    matR = (matP) / (sums[None, :])
    return(matR)


def createGMatrix(matR):
    sums = np.sum(matR, axis=1)
    matG = np.diag(sums)
    return(matG)


def optimWMatrix(matR, matPhiMPlusOne, matG, data, betaInv, l):
    n_rbf_centersP = matPhiMPlusOne.shape[1]
    LBmat = np.zeros([n_rbf_centersP, n_rbf_centersP])
    PhiGPhi = np.dot(
        np.dot(np.transpose(matPhiMPlusOne), matG), matPhiMPlusOne)
    for i in range(n_rbf_centersP):
        LBmat[i][i] = l * betaInv
    PhiGPhiLB = PhiGPhi + LBmat
    Ginv = np.linalg.inv(PhiGPhiLB)
    matW = np.transpose(
        np.dot(np.dot(np.dot(Ginv, np.transpose(matPhiMPlusOne)), matR), data))
    return(matW)


def optimLMatrix(matR, matPhiMPlusOne, matG, betaInv, l):
    n_rbf_centersP = matPhiMPlusOne.shape[1]
    LBmat = np.zeros([n_rbf_centersP, n_rbf_centersP])
    PhiGPhi = np.dot(
        np.dot(np.transpose(matPhiMPlusOne), matG), matPhiMPlusOne)
    for i in range(n_rbf_centersP):
        LBmat[i][i] = l * betaInv
    PhiGPhiLB = PhiGPhi + LBmat
    Ginv = np.linalg.inv(PhiGPhiLB)
    matW = np.transpose(
        np.dot(np.dot(Ginv, np.transpose(matPhiMPlusOne)), matR))
    return(matW)


def optimBetaInv(matR, matD, n_dimensions):
    n_individuals = matR.shape[1]
    betaInv = np.sum(np.multiply(matR, matD))/(n_individuals*n_dimensions)
    return(betaInv)


def meanPoint(matR, matX):
    matMeans = np.dot(np.transpose(matR), matX)
    return(matMeans)


def modePoint(matR, matX):
    matModes = matX[np.argmax(matR, axis=0), :]
    return(matModes)


def computelogLikelihood(matP, betaInv, n_dimensions):
    LogLikelihood = np.longdouble()
    n_nodes = matP.shape[0]
    n_individuals = matP.shape[1]
    LogLikelihood = 0.0
    prior = 1.0/n_nodes
    placeholder = 50
    constante = np.longdouble(np.power(((1/betaInv)/(2*np.pi)),
                                       np.minimum(
                                        n_dimensions/2, placeholder)))
    LogLikelihood = np.sum(np.log(
                            np.maximum(np.sum(constante*matP, axis=0)*prior,
                                       np.finfo(np.longdouble).tiny)))
    LogLikelihood /= n_individuals
    return(-LogLikelihood)


def evalBetaInv(matY, betaInv, random_state=1234):
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
    betaInv = np.sum(matD*1/n_nodes)/(n_individuals*n_dimensions)
    return(betaInv)
