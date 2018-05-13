from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from .ugtm_preprocess import *
from .ugtm_classes import *
from .ugtm_core import *

def initialize(data,k,m,s,random_state=1234):
	n_individuals = data.shape[0]
	n_dimensions = data.shape[1]
	n_nodes = k*k
	n_rbf_centers = m*m
	x = np.linspace(-1, 1, k)
	matX = np.transpose(np.meshgrid(x,x)).reshape(k*k,2)
	x = np.linspace(-1, 1, m)
	matM = np.transpose(np.meshgrid(x,x)).reshape(m*m,2)
	rbfWidth = computeWidth(matM,n_rbf_centers,s) 
	matPhiMPlusOne = createPhiMatrix(matX,matM,n_nodes,n_rbf_centers,rbfWidth)
	pca = PCA(n_components=3,random_state=random_state)
	pca.fit(data)
	matU=(pca.components_.T * np.sqrt(pca.explained_variance_))[:,0:2]
	betaInv=pca.explained_variance_[2]
	Uobj = ReturnU(matU,betaInv)
	matW = createWMatrix(matX,matPhiMPlusOne,Uobj.matU,n_dimensions,n_rbf_centers)
	matY = createYMatrixInit(data,matW,matPhiMPlusOne)
	betaInv = Uobj.betaInv
	betaInv = evalBetaInv(matY,Uobj.betaInv,random_state=random_state) 
	return InitialGTM(matX, matM, n_nodes, n_rbf_centers, rbfWidth, matPhiMPlusOne, \
                          matW, matY, betaInv,n_dimensions)

def optimize(data, initialModel, l, niter, verbose=True):
	matD = createDistanceMatrix(initialModel.matY, data)
	matY = initialModel.matY
	betaInv = initialModel.betaInv
	i = 1
	diff = 1000
	converged = 0
	while i<(niter+1) and (converged<4):	
		#expectation
		matP = createPMatrix(matD,betaInv,initialModel.n_dimensions)
		matR=createRMatrix(matP)
		#maximization
		matG = createGMatrix(matR)
		matW = optimWMatrix(matR, initialModel.matPhiMPlusOne, matG, data, betaInv, l)
		matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
		matD = createDistanceMatrix(matY, data)
		betaInv = optimBetaInv(matR,matD,initialModel.n_dimensions)
		#objective function
		if i == 1:
			loglike = computelogLikelihood(matP,betaInv,initialModel.n_dimensions);
		else:
			loglikebefore=loglike
			loglike = computelogLikelihood(matP,betaInv,initialModel.n_dimensions)
			diff=abs(loglikebefore-loglike)
		if diff <= 0.0001:
			converged += 1
		else:
			converged = 0
		if verbose:
			#print("Iter ", i, " Err: ", loglike," Beta: ",1/betaInv)
			print("Iter ", i, " Err: ", loglike)
		i += 1
	#final iteration to make sure matR fits matD
	if verbose == 1:
		if converged >= 3:
			print("Converged: ", loglike)
	if converged >= 3:
		has_converged = True
	else:
		has_converged = False
	matP = createPMatrix(matD,betaInv,initialModel.n_dimensions)
	matR = createRMatrix(matP)
	matMeans = meanPoint(matR, initialModel.matX)
	matModes = modePoint(matR, initialModel.matX)
	return OptimizedGTM(matW, matY, matP.T, matR.T, betaInv, matMeans, matModes, \
                            initialModel.matX,initialModel.n_dimensions,has_converged)

def projection(optimizedModel,new_data):
	matD = createDistanceMatrix(optimizedModel.matY, new_data)
	matP = createPMatrix(matD,optimizedModel.betaInv,optimizedModel.n_dimensions)
	matR = createRMatrix(matP)
	loglike = computelogLikelihood(matP,optimizedModel.betaInv,optimizedModel.n_dimensions)
	matMeans = meanPoint(matR, optimizedModel.matX)
	matModes = modePoint(matR, optimizedModel.matX)
	return OptimizedGTM(optimizedModel.matW, optimizedModel.matY, matP.T, matR.T, \
                            optimizedModel.betaInv, matMeans, matModes, \
                            optimizedModel.matX,optimizedModel.n_dimensions, \
                            optimizedModel.converged)

def transform(optimizedModel,train,test,doPCA=False,n_components=-1,missing=True, \
              missing_strategy="median",random_state=1234,process=True):
	if process==True:
		processed = processTrainTest(train, test, doPCA=doPCA, \
                                             n_components=n_components,missing=missing, \
                                             missing_strategy=missing_strategy, \
                                             random_state=random_state)
	if process==True:
		return projection(optimizedModel,processed.test)
	else:
		return projection(optimizedModel,test)


def runGTM(data,doPCA=False,n_components=-1,missing=True,missing_strategy="median", \
           random_state=1234,k=16,m=4,s=0.3,l=0.1,niter=200,verbose=False):
	data=pcaPreprocess(data=data,doPCA=doPCA,n_components=n_components, \
                           missing=missing,missing_strategy=missing_strategy, \
                           random_state=random_state)
	initialModel = initialize(data,k,m,s,random_state)
	optimizedModel = optimize(data,initialModel,l,niter,verbose=verbose)
	return optimizedModel
