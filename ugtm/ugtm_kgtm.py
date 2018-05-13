from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from .ugtm_preprocess import *
from .ugtm_classes import *
from .ugtm_core import *

def initializeKernel(data, k, m, s, maxdim,random_state=1234):
	n_individuals = data.shape[0]
	n_nodes = k*k
	n_rbf_centers = m*m
	x = np.linspace(-1, 1, k)
	matX = np.transpose(np.meshgrid(x,x)).reshape(k*k,2)
	x = np.linspace(-1, 1, m)
	matM = np.transpose(np.meshgrid(x,x)).reshape(m*m,2)
	rbfWidth = computeWidth(matM,n_rbf_centers,s)
	matPhiMPlusOne = createPhiMatrix(matX,matM,n_nodes,n_rbf_centers,rbfWidth)
	pca = PCA(random_state=random_state)
	pca.fit(data)
	matW = (pca.components_.T * np.sqrt(pca.explained_variance_))[:,0:n_rbf_centers+1]
	n_dimensions = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.995)+1
	if n_dimensions > maxdim:
		n_dimensions = maxdim 
	#betaInv=pca.explained_variance_[2]
	matD = KERNELcreateDistanceMatrix(data, matW, matPhiMPlusOne)
	betaInv = initBetaInvRandom(matD,n_nodes,n_individuals,n_dimensions)
	matY = createYMatrixInit(data,matW,matPhiMPlusOne)
	return InitialGTM(matX, matM, n_nodes, n_rbf_centers, rbfWidth, \
                          matPhiMPlusOne, matW, matY, betaInv,n_dimensions)

def optimizeKernel(data, initialModel, l, niter,verbose=True):
	matD = KERNELcreateDistanceMatrix(data, initialModel.matW, initialModel.matPhiMPlusOne)
	matY = initialModel.matY
	betaInv = initialModel.betaInv
	i = 1
	diff = 1000
	converged = 0
	while (i<(niter+1)) and (converged<4):
		#expectation
		matP = createPMatrix(matD,betaInv,initialModel.n_dimensions)
		matR = createRMatrix(matP)
		#maximization
		matG = createGMatrix(matR)
		matW = optimLMatrix(matR, initialModel.matPhiMPlusOne, matG, betaInv, l)
		matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
		matD = KERNELcreateDistanceMatrix(data, matW, initialModel.matPhiMPlusOne)
		betaInv = optimBetaInv(matR,matD,initialModel.n_dimensions)
		#objective function
		if i == 1:
			loglike = computelogLikelihood(matP,betaInv,initialModel.n_dimensions);
		else:
			loglikebefore = loglike
			loglike = computelogLikelihood(matP,betaInv,initialModel.n_dimensions)
			diff = abs(loglikebefore-loglike)
		if verbose == True:
			print("Iter ", i, " Err: ", loglike)
		if diff <= 0.0001:
			converged+=1
		else:
			converged=0
		i += 1
	if verbose == True:
		if converged >= 3: 
			print("Converged: ",loglike)
	if converged >= 3:
		has_converged = True
	else:
		has_converged = False
	matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
	matMeans = meanPoint(matR, initialModel.matX)
	matModes = modePoint(matR, initialModel.matX)
	return OptimizedGTM(matW, matY, matP.T, matR.T, betaInv, matMeans, matModes, \
                            initialModel.matX, initialModel.n_dimensions,has_converged)

def runkGTM(data,doPCA=False,doKernel=False,kernel="linear",n_components=-1, \
            maxdim=100,missing=True,missing_strategy="median", \
            random_state=1234,k=16,m=4,s=0.3,l=0.1,niter=200,verbose=False):
	data=pcaPreprocess(data=data,doPCA=doPCA,n_components=n_components, \
                           missing=missing,missing_strategy=missing_strategy, \
                           random_state=random_state)
	if doKernel or data.shape[0]!=data.shape[1]:
		data = chooseKernel(data,kernel)
	initialModel = initializeKernel(data,k,m,s,random_state)
	optimizedModel = optimizeKernel(data,initialModel,l,niter,verbose=verbose)
	return optimizedModel
