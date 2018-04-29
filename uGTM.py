from __future__ import print_function
import pandas as pd
import numpy as np
import os
import scipy
import sys
import math
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KernelCenterer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold 
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import random
from scipy.interpolate import Rbf
from scipy import interpolate
from matplotlib import cm
import scipy.stats as st

#create manifold
def createYMatrixInit(matT,matW,matPhiMPlusOne):
	shap1 = matW.shape[0];
	shap2 = matPhiMPlusOne.shape[0];
	TheMeans = matT.mean(0)
	DMmeanMatrix = np.zeros([shap1, shap2])
	for i in range(shap1):
		for j in range(shap2):
			DMmeanMatrix[i,j] = TheMeans[i]
	MatY = np.dot(matW, np.transpose(matPhiMPlusOne))
	MatY = MatY + DMmeanMatrix
	return(MatY)

def createPhiMatrix(matX,matM,numX, numM,sigma):
	Result = np.zeros([numX, numM + 1])
	for i in range(numX):
		for j in range(numM):
			Coo1 = (matX[i][0] - matM[j][0]) * (matX[i][0] - matM[j][0])
			Coo2 = (matX[i][1] - matM[j][1]) * (matX[i][1] - matM[j][1])
			Distance = Coo1 + Coo2
			Result[i,j] = np.exp(-(Distance) / (2 * sigma))
	for i in range(numX):
		Result[i][numM] = 1
	return(Result)


def computeWidth(matM,numM,sigma):
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


def createFeatureMatrixNIPALS(matT, nMolecules, nDimensions,random_state=1234):
	r = np.random.RandomState(random_state)
	Result = np.zeros([nDimensions, 2])
	threshold = 0.0001
	id = 0
#shortest way: take first column
	t = matT[:,id]
	if (np.sum(t) == 0):
		for i in range(nMolecules):
			t[i] = abs(r.uniform(-1, 1, size=1))
	for i in range(2):
		tauold = 160
		taunew = 0 
		while (abs(tauold-taunew)>(threshold)):
			tauold = np.dot(t,t)
			if tauold == 0.0:
				tauold = sys.float_info.min
			p = np.dot(t,matT)*(1/tauold)
			pFloat = np.dot(p,p)
			if pFloat == 0.0:
				pFloat = sys.float_info.min
			p = p*(1/math.sqrt(pFloat))
			pFloat = np.dot(p,p)
			if pFloat == 0.0:
				pFloat = sys.float_info.min
			t = np.dot(matT,p)*(1/pFloat)
			taunew = np.dot(t,t)
		pc = np.outer(t,p)
		matT = matT - pc
#the U matrix is made up of loadings,
#which are eigenvectors scaled by the sqrt of eigenvalues taunew
#we divide the eigenvalues by the number of observations
#to match the SVD covariance matrix method results (covariance matrix = 1/(n-1)*(T'T) )
		for j in range(nDimensions):
			Result[j][i] = p[j]*(math.sqrt(taunew/nMolecules))

	for j in range(nDimensions):
		Result[j][1] = - Result[j][1]
# compute third eigenvalue: perform supplementary iteration of NIPALS
	tauold = 160
	taunew = 0
	while (abs(tauold-taunew)>(threshold)):
		tauold = np.dot(t,t)
		if tauold == 0.0:
			tauold = sys.float_info.min
		p = np.dot(t,matT)*(1/tauold)
		pFloat = np.dot(p,p)
		if pFloat == 0.0:
			pFloat = sys.float_info.min
		p = p*(1/math.sqrt(pFloat))
		pFloat = np.dot(p,p)
		if pFloat == 0.0:
			pFloat = sys.float_info.min
		t = np.dot(matT,p)*(1/pFloat)
		taunew = np.dot(t,t)
#inverse variance is initialized as the 3rd eigenvalue
	betaInv = taunew/nMolecules

	return ReturnU(Result,betaInv)


def createWMatrix(matX, matPhiMPlusOne, matU, nDimensions, nCenters):
	NormX = preprocessing.scale(matX,axis=0, with_mean=True, with_std=True) 
	myProd = np.dot(matU,np.transpose(NormX))
	tinv = np.linalg.solve(matPhiMPlusOne.T.dot(matPhiMPlusOne), matPhiMPlusOne.T)
	Result = np.dot(myProd,np.transpose(tinv))
	return(Result)


def createYMatrix(matW,matPhiMPlusOne):
	Result = np.dot(matW, np.transpose(matPhiMPlusOne))
	return(Result)


class ReturnExtendedR(object):
	def __init__(self, matR, ids):
		self.matR = matR 
		self.ids = ids

class ReturnProcessedTrainTest(object):
	def __init__(self, train, test):
		self.train = train 
		self.test = test

class ReturnClassMap(object):
	def __init__(self, nodeClassP, nodeClassT, activityModel, uniqClasses):
		self.nodeClassP = nodeClassP
		self.nodeClassT = nodeClassT
		self.activityModel = activityModel
		self.uniqClasses = uniqClasses


class ReturnInitial(object):
	def __init__(self, matX, matM, nSamples, nCenters, rbfWidth, matPhiMPlusOne, matW, matY,betaInv,nDimensions):
		self.matX = matX 
		self.matM = matM 
		self.nCenters = nCenters
		self.nSamples = nSamples
		self.rbfWidth = rbfWidth
		self.matPhiMPlusOne = matPhiMPlusOne
		self.matW = matW
		self.matY = matY
		self.betaInv = betaInv
		self.nDimensions = nDimensions

class ReturnOptimized(object):
	def __init__(self, matW, matY, matP, matR, betaInv, matMeans, matModes):
		self.matW = matW
		self.matY = matY
		self.matP = matP
		self.matR = matR
		self.betaInv = betaInv
		self.matMeans = matMeans
		self.matModes = matModes

class ReturnU(object):
	def __init__(self, matU, betaInv):
		self.matU = matU 
		self.betaInv = betaInv 


def initializeKernel(matT, k, m, s, maxdim,random_state=1234):
	nMolecules = matT.shape[0]
	nSamples = k*k
	nCenters = m*m
	x = np.linspace(-1, 1, k)
	matX = np.transpose(np.meshgrid(x,x)).reshape(k*k,2)
	x = np.linspace(-1, 1, m)
	matM = np.transpose(np.meshgrid(x,x)).reshape(m*m,2)
	rbfWidth = computeWidth(matM,nCenters,s)
	matPhiMPlusOne = createPhiMatrix(matX,matM,nSamples,nCenters,rbfWidth)
	pca = PCA(random_state=random_state)
	pca.fit(matT)
	matW = (pca.components_.T * np.sqrt(pca.explained_variance_))[:,0:nCenters+1]
	nDimensions = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.995)+1
	if nDimensions > maxdim:
		nDimensions = maxdim 
	#betaInv=pca.explained_variance_[2]
	matD = KERNELcreateDistanceMatrix(matT, matW, matPhiMPlusOne)
	betaInv = initBetaInvRandom(matD,nSamples,nMolecules,nDimensions)
	matY = createYMatrixInit(matT,matW,matPhiMPlusOne)
	return ReturnInitial(matX, matM, nSamples, nCenters, rbfWidth, matPhiMPlusOne, matW, matY, betaInv,nDimensions)

def optimizeKernel(matT, initialModel, alpha, niter,verbose=True):
	matD = KERNELcreateDistanceMatrix(matT, initialModel.matW, initialModel.matPhiMPlusOne)
	matY = initialModel.matY
	betaInv = initialModel.betaInv
	i = 1
	diff = 1000
	converged = 0
	while (i<(niter+1)) and (converged<4):
		#expectation
		matP = createPMatrix(matD,betaInv,initialModel.nDimensions)
		matR = createRMatrix(matP)
		#maximization
		matG = createGMatrix(matR)
		matW = optimLMatrix(matR, initialModel.matPhiMPlusOne, matG, betaInv, alpha)
		matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
		matD = KERNELcreateDistanceMatrix(matT, matW, initialModel.matPhiMPlusOne)
		betaInv = optimBetaInv(matR,matD,initialModel.nDimensions)
		#objective function
		if i == 1:
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions);
		else:
			loglikebefore = loglike
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions)
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
	matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
	matMeans = meanPoint(matR, initialModel.matX)
	matModes = modePoint(matR, initialModel.matX)
	return ReturnOptimized(matW, matY, matP.T, matR.T, betaInv, matMeans, matModes)


def landscape(optimizedModel,activity):
	epsilon=10e-8
	sums = np.sum(optimizedModel.matR+epsilon,axis=0)
	landscape = np.dot(activity,optimizedModel.matR+epsilon) / sums[None,:]
	return np.asarray(landscape)[0,:]

def classMap(optimizedModel,activity,prior="equiprobable"):
	uniqClasses, classVector = np.unique(activity, return_inverse=True)
	nClasses = uniqClasses.shape[0]
	nMolecules = optimizedModel.matR.shape[0]
	nSamples = optimizedModel.matR.shape[1]
	#posterior distribution
	nodeClassP = np.zeros([nSamples,nClasses])
	#likelihood
	nodeClassT = np.zeros([nSamples,nClasses])
	sumClass = np.zeros([nClasses])
	summe = np.zeros([nSamples])
	for i in range(nClasses):
	    sumClass[i]=(classVector==i).sum()
	if prior=="estimated":
	    priors=sumClass/sumClass.sum()
	elif prior=="equiprobable":
	    priors=np.zeros([nClasses])+(1.0/nClasses)

	for i in range(nClasses):
		for k in range(nSamples):
			nodeClassT[k,i] = optimizedModel.matR[classVector==i,k].sum()/sumClass[i]

	for i in range(nClasses):
		for k in range(nSamples):
			nodeClassP[k,i]=nodeClassT[k,i]*priors[i]
			summe[k]+=nodeClassP[k,i]


	for i in range(nClasses):
		for k in range(nSamples):
			if summe[k]!=0.0:
				nodeClassP[k,i]=nodeClassP[k,i]/summe[k]

	for k in range(nSamples):
		if summe[k] == 0.0:
			for i in range(nClasses):
				nodeClassP[k,i]=1/nClasses
	
	nodeClass = np.argmax(nodeClassP, axis=1)
	return(ReturnClassMap(nodeClassP,nodeClassT,nodeClass,uniqClasses))


def initialize(matT,k,m,s,random_state=1234):
	nMolecules = matT.shape[0]
	nDimensions = matT.shape[1]
	nSamples = k*k
	nCenters = m*m
	x = np.linspace(-1, 1, k)
	matX = np.transpose(np.meshgrid(x,x)).reshape(k*k,2)
	x = np.linspace(-1, 1, m)
	matM = np.transpose(np.meshgrid(x,x)).reshape(m*m,2)
	rbfWidth = computeWidth(matM,nCenters,s) 
	matPhiMPlusOne = createPhiMatrix(matX,matM,nSamples,nCenters,rbfWidth)
	Uobj = createFeatureMatrixNIPALS(matT, nMolecules, nDimensions,random_state=random_state)
	#alternative for creating U loading matrix: instead of NIPALS, use PCA (it's slower.....):
##	pca = PCA(n_components=3)
##	pca.fit(matT)
##	matU=(pca.components_.T * np.sqrt(pca.explained_variance_))[:,0:2]
##	betaInv=pca.explained_variance_[2]
	matW = createWMatrix(matX,matPhiMPlusOne,Uobj.matU,nDimensions,nCenters)
	matY = createYMatrixInit(matT,matW,matPhiMPlusOne)
	betaInv = Uobj.betaInv
	betaInv = evalBetaInv(matY,Uobj.betaInv,random_state=random_state) 
	return ReturnInitial(matX, matM, nSamples, nCenters, rbfWidth, matPhiMPlusOne, matW, matY, betaInv,nDimensions)


def createDistanceMatrix(matY, matT):
	Result = scipy.spatial.distance.cdist(matY.T,matT,metric='sqeuclidean')	
	return(Result)

def exp_normalize(x):
    y = np.array([],dtype=np.longdouble)
    y = x - np.expand_dims(np.max(x, axis = 0), 0)
    y = np.exp(y)
    return(y)

def createPMatrix(matD,betaInv,nDimensions):
	matP = np.array([],dtype=np.longdouble) 
	beta = 1/betaInv
	dim = nDimensions
	#constante = np.power(((beta)/(2*np.pi)),dim/2)
	constante = 1
	matP = exp_normalize(-(beta/2)*matD)
	return(matP)


def createRMatrix(matP):
	matR = np.array([],dtype=np.longdouble)
	sums = np.sum(matP,axis=0)
	matR = (matP) / (sums[None,:])
	return(matR)

def optimize(matT, initialModel, alpha, niter, verbose=True):
	matD = createDistanceMatrix(initialModel.matY, matT)
	matY = initialModel.matY
	betaInv = initialModel.betaInv
	i = 1
	diff = 1000
	converged = 0
	while i<(niter+1) and (converged<4):	
		#expectation
		matP = createPMatrix(matD,betaInv,initialModel.nDimensions)
		matR=createRMatrix(matP)
		#maximization
		matG = createGMatrix(matR)
		matW = optimWMatrix(matR, initialModel.matPhiMPlusOne, matG, matT, betaInv, alpha)
		matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
		matD = createDistanceMatrix(matY, matT)
		betaInv = optimBetaInv(matR,matD,initialModel.nDimensions)
		#objective function
		if i == 1:
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions);
		else:
			loglikebefore=loglike
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions)
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
	matP = createPMatrix(matD,betaInv,initialModel.nDimensions)
	matR = createRMatrix(matP)
	matMeans = meanPoint(matR, initialModel.matX)
	matModes = modePoint(matR, initialModel.matX)
	return ReturnOptimized(matW, matY, matP.T, matR.T, betaInv, matMeans, matModes)

def createGMatrix(matR):
	sums = np.sum(matR,axis=1)
	matG = np.diag(sums)
	return(matG) 

def optimWMatrix(matR, matPhiMPlusOne, matG, matT, betaInv, alpha):
	nCentersP = matPhiMPlusOne.shape[1]
	LBmat = np.zeros([nCentersP, nCentersP])
	PhiGPhi = np.dot(np.dot(np.transpose(matPhiMPlusOne),matG), matPhiMPlusOne)
	for i in range(nCentersP):
		LBmat[i][i] = alpha * betaInv
	PhiGPhiLB = PhiGPhi + LBmat
	Ginv = np.linalg.inv(PhiGPhiLB)
	matW = np.transpose(np.dot(np.dot(np.dot(Ginv, np.transpose(matPhiMPlusOne)),matR),matT))
	return(matW)

def optimLMatrix(matR, matPhiMPlusOne, matG, betaInv, alpha):
	nCentersP = matPhiMPlusOne.shape[1]
	LBmat = np.zeros([nCentersP, nCentersP])
	PhiGPhi = np.dot(np.dot(np.transpose(matPhiMPlusOne),matG), matPhiMPlusOne)
	for i in range(nCentersP):
		LBmat[i][i] = alpha * betaInv
	PhiGPhiLB = PhiGPhi + LBmat
	Ginv = np.linalg.inv(PhiGPhiLB)
	matW = np.transpose(np.dot(np.dot(Ginv, np.transpose(matPhiMPlusOne)),matR))
	return(matW)

def optimBetaInv(matR,matD,nDimensions):
	nMolecules = matR.shape[1]
	betaInv = np.sum(np.multiply(matR,matD))/(nMolecules*nDimensions)
	return(betaInv)

def meanPoint(matR,matX):
	matMeans = np.dot(np.transpose(matR), matX)
	return(matMeans)

def modePoint(matR,matX):
	matModes = matX[np.argmax(matR, axis=0),:]
	return(matModes)

def computelogLikelihood(matP,betaInv,nDimensions):
	LogLikelihood=np.longdouble()
	nSamples = matP.shape[0]
	nMolecules = matP.shape[1]
	Ptwb = 0.0
	LogLikelihood = 0.0
	prior = 1.0/nSamples
	placeholder=50
	constante=np.longdouble(np.power(((1/betaInv)/(2*np.pi)),np.minimum(nDimensions/2,placeholder)))
	LogLikelihood = np.sum(np.log(np.maximum(np.sum(constante*matP,axis=0)*prior,np.finfo(np.longdouble).tiny)))
	LogLikelihood /= nMolecules
	return(-LogLikelihood)

def evalBetaInv(matY,betaInv,random_state=1234):
	r = np.random.RandomState(random_state)
	Distances = sklearn.metrics.pairwise.euclidean_distances(np.transpose(matY))
	myMin = np.mean(Distances)/2;
	myMin *= myMin;
	if((myMin<betaInv) or (betaInv==0)):
		betaInv = myMin
	if betaInv == 0.0:
		print('bad initialization (0 variance), setting variance to random number...');
		betaInv = abs(r.uniform(-1, 1, size=1));
	return(betaInv);


def KERNELcreateDistanceMatrix(matT, matL, matPhiMPlusOne):
	nSamples = matPhiMPlusOne.shape[0]
	nMolecules = matT.shape[0]
	Result = np.zeros([nSamples, nMolecules])
	thefloat = 0.0
	for i in range(nSamples):
		LPhim = np.dot(matL, matPhiMPlusOne[i])
		thefloat = np.dot(np.dot(LPhim,matT),LPhim)
		for j in range(nMolecules):
			Result[i,j] = matT[j,j] + thefloat - 2*(np.dot(matT[j],LPhim))
	return(Result)

def initBetaInvRandom(matD,nSamples,nMolecules,nDimensions):
	betaInv = np.sum(matD*1/nSamples)/(nMolecules*nDimensions)
	return(betaInv)


def plotLandscape(initialModel,optimizedModel,label):
	k = math.sqrt(initialModel.nSamples);
	x = initialModel.matX[:,0]
	y = initialModel.matX[:,1]
	z = landscape(optimizedModel,label)
	ti = np.linspace(-1.0, 1.0, k)
	XI, YI = np.meshgrid(ti, ti)
	f = interpolate.NearestNDInterpolator(initialModel.matX,z)
	ZI=f(XI,YI)
	plt.pcolor(XI, YI, ZI, cmap=plt.cm.Spectral)
	plt.scatter(x, y, 50*(10/k), z, cmap=plt.cm.Spectral,marker="s")
	plt.scatter(optimizedModel.matMeans[:,0], optimizedModel.matMeans[:,1], 20, label, cmap=plt.cm.Spectral,edgecolor='black',marker="o")
	plt.title('Landscape')
	plt.xlim(-1.1, 1.1)
	plt.ylim(-1.1, 1.1)
	plt.colorbar()
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])


def plotLandscapeNoPoints(initialModel,optimizedModel,label):
	k = math.sqrt(initialModel.nSamples);
	x = initialModel.matX[:,0]
	y = initialModel.matX[:,1]
	z = landscape(optimizedModel,label)
	ti = np.linspace(-1.0, 1.0, k)
	XI, YI = np.meshgrid(ti, ti)
	f = interpolate.NearestNDInterpolator(initialModel.matX,z)
	ZI=f(XI,YI)
	plt.pcolor(XI, YI, ZI, cmap=plt.cm.Spectral)
	plt.scatter(x, y, 50*(10/k), z, cmap=plt.cm.Spectral,marker="s")
	plt.title('Landscape')
	plt.xlim(-1.1, 1.1)
	plt.ylim(-1.1, 1.1)
	plt.colorbar()
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])

def plotClassMap(initialModel,optimizedModel,label,prior="equiprobable"):
	k = math.sqrt(initialModel.nSamples);
	x = initialModel.matX[:,0]
	y = initialModel.matX[:,1]
	z = classMap(optimizedModel,label,prior).activityModel
	#ti = np.linspace(-1.0, 1.0, k)
	#XI, YI = np.meshgrid(ti, ti)
	#f = interpolate.NearestNDInterpolator(initialModel.matX,z)
	#ZI=f(XI,YI)
	#plt.pcolor(XI, YI, ZI, cmap=plt.cm.Spectral)
	uniqClasses, label = np.unique(label, return_inverse=True)
	plt.scatter(x, y, 175*(10/k), z, cmap=plt.cm.Spectral,marker="s",alpha=0.3)
	plt.scatter(optimizedModel.matMeans[:,0], optimizedModel.matMeans[:,1], 20, label, cmap=plt.cm.Spectral,edgecolor='black',marker="o")
	#plt.scatter(optimizedModel.matModes[:,0], optimizedModel.matModes[:,1], 20, c="blue", edgecolor='black',marker="o")
	plt.title('Class Map')
	plt.xlim(-1.1, 1.1)
	plt.ylim(-1.1, 1.1)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])	


def plotClassMapNoPoints(initialModel,optimizedModel,label,prior="equiprobable"):
	k = math.sqrt(initialModel.nSamples);
	x = initialModel.matX[:,0]
	y = initialModel.matX[:,1]
	z = classMap(optimizedModel,label,prior).activityModel
	#ti = np.linspace(-1.0, 1.0, k)
	#XI, YI = np.meshgrid(ti, ti)
	#f = interpolate.NearestNDInterpolator(initialModel.matX,z)
	#ZI=f(XI,YI)
	#plt.pcolor(XI, YI, ZI, cmap=plt.cm.Spectral)
	uniqClasses, label = np.unique(label, return_inverse=True)
	plt.scatter(x, y, 175*(10/k), z, cmap=plt.cm.Spectral,marker="s",alpha=0.3)
	plt.title('Class Map')
	plt.xlim(-1.1, 1.1)
	plt.ylim(-1.1, 1.1)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])

def projection(initialModel,optimizedModel,newT):
	matD = createDistanceMatrix(optimizedModel.matY, newT)
	matP = createPMatrix(matD,optimizedModel.betaInv,initialModel.nDimensions)
	matR = createRMatrix(matP)
	loglike = computelogLikelihood(matP,optimizedModel.betaInv,initialModel.nDimensions)
	matMeans = meanPoint(matR, initialModel.matX)
	matModes = modePoint(matR, initialModel.matX)
	return ReturnOptimized(optimizedModel.matW, optimizedModel.matY, matP.T, matR.T, optimizedModel.betaInv, matMeans, matModes)

def predictNN(initialModel,optimizedModel,labels,newT,modeltype,n_neighbors=1,representation="modes",prior="equiprobable"):
	if modeltype=='regression':
		activityModel=landscape(optimizedModel,labels)
	elif modeltype=='classification':
		n_neibhbors=1
		activityModel=classMap(optimizedModel,labels,prior).activityModel
	projected = projection(initialModel,optimizedModel,newT)
	neighborModel = NearestNeighbors(n_neighbors=n_neighbors,metric='euclidean')
	fitted = neighborModel.fit(initialModel.matX)
	if representation=='means':
		rep=projected.matMeans
	elif representation=='modes':	
		rep=projected.matModes
	if modeltype=='regression' and n_neighbors>1:
		dist,nnID = fitted.kneighbors(rep,return_distance=True)
		dist[dist<=0]=np.finfo(float).tiny
		predicted = np.average(activityModel[nnID],axis=1,weights=1/((dist)**2))
	else:
		nnID = fitted.kneighbors(rep,return_distance=False)
		predicted=activityModel[nnID]
	return predicted


def predictNNSimple(train,test,labels,n_neighbors=1,modeltype='regression'):
	if modeltype=='regression' and n_neighbors>1:
		neighborModel = NearestNeighbors(n_neighbors=n_neighbors,metric='euclidean')
		fitted = neighborModel.fit(train)
		dist,nnID = fitted.kneighbors(test,return_distance=True)
		dist[dist<=0]=np.finfo(float).tiny
		predicted = np.average(labels[nnID],axis=1,weights=1/((dist)**2))
	
	else:
		clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
		clf.fit(train,labels)
		predicted=clf.predict(test)
	return predicted



def predictBayes(initialModel,optimizedModel,labels,newT,prior="equiprobable"):
	activityModel=classMap(optimizedModel,labels,prior).nodeClassP
	projected = projection(initialModel,optimizedModel,newT).matR
	predicted=np.argmax(np.matmul(projected,activityModel),axis=1)
	return predicted

def crossvalidatePCAC(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,n_neighbors=1):
	if n_components == -1 and doPCA == True:
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	modeltype = "classification"
	uniqClasses, labels = np.unique(labels, return_inverse=True)
	nClasses=len(uniqClasses)
	print("Classes: ",uniqClasses)
	print("nClasses: ",nClasses)
	print("")
	print("model\tparameters=k_for_kNN\tavg. weighted recall with CI\t avg. weighted precision with CI\t avg. weighted F1-score with CI",end="")
	for i in range(nClasses): 
		print("\trecall %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tprecision %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tF1-score %s" % (uniqClasses[i]), end="")
	print("")
	if n_neighbors <= 0:
		Kvec = np.arange(start=1,stop=31,step=1,dtype=np.int32)
	else:
		Kvec = [ n_neighbors ]

	savemean = -9999 
	saveh = 0.0
	nummodel = 0
	savemodel = ""
	for c in Kvec:
		nummodel += 1
		modelstring = str(c)
		recallvec = []
		precisionvec = []
		f1vec = []
		recallclassvec = np.array([])
		precisionclassvec = np.array([])
		f1classvec = np.array([])
		meanclass=np.zeros(nClasses)
		meanprecisionclass = np.zeros(nClasses)
		meanf1class = np.zeros(nClasses)
		seclass=np.zeros(nClasses)
		seprecisionclass=np.zeros(nClasses)
		sef1class=np.zeros(nClasses)
		hclass=np.zeros(nClasses)
		hprecisionclass=np.zeros(nClasses)
		hf1class=np.zeros(nClasses)
		for j in range(10):
			ss = KFold(n_splits=5, shuffle=True, random_state=j)
			y_true=[]
			y_pred=[]
			for train_index, test_index in ss.split(matT):
				train=np.copy(matT[train_index])
				test=np.copy(matT[test_index])
				processed=processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
				y_pred = np.append(y_pred,predictNNSimple(processed.train,processed.test,labels[train_index],c,"classification"))
				y_true = np.append(y_true,labels[test_index])
			recall = recall_score(y_true, y_pred, average='weighted')
			precision = precision_score(y_true, y_pred, average='weighted')
			f1 = f1_score(y_true, y_pred, average='weighted')
			recallvec = np.append(recallvec,recall)
			precisionvec = np.append(precisionvec,precision)
			f1vec = np.append(f1vec,f1)
			recallclass = recall_score(y_true,y_pred,average=None)
			precisionclass = precision_score(y_true,y_pred,average=None)
			f1class =  f1_score(y_true,y_pred,average=None)
			if(j==0):
				recallclassvec = recallclass
				precisionclassvec = precisionclass
				f1classvec = f1class
			else:
				recallclassvec = np.vstack([recallclassvec,recallclass])
				precisionclassvec =  np.vstack([precisionclassvec,precisionclass])
				f1classvec =  np.vstack([f1classvec,f1class])
		mean, se = np.mean(recallvec), st.sem(recallvec)
		meanprecision, seprecision = np.mean(precisionvec), st.sem(precisionvec)
		meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
		h = se * scipy.stats.t._ppf((1+0.95)/2., len(recallvec)-1)
		hprecision = seprecision * scipy.stats.t._ppf((1+0.95)/2., len(precisionvec)-1)
		hf1 = sef1 * scipy.stats.t._ppf((1+0.95)/2., len(f1vec)-1)
		if(meanf1 > savemean):
			savemean=meanf1
			saveh=hf1
			modelvec=modelstring
			savemodel="Model "+str(nummodel)
		for i in range(0,nClasses):
			meanclass[i], seclass[i] = np.mean(recallclassvec[:,i]), st.sem(recallclassvec[:,i])
			meanf1class[i], sef1class[i] = np.mean(f1classvec[:,i]), st.sem(f1classvec[:,i])
			meanprecisionclass[i], seprecisionclass[i] = np.mean(precisionclassvec[:,i]), st.sem(precisionclassvec[:,i])
			hclass[i] =  seclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(recallclassvec[:,i])-1)
			hprecisionclass[i] =  seprecisionclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(precisionclassvec[:,i])-1)
			hf1class[i] =  sef1class[i] * scipy.stats.t._ppf((1+0.95)/2., len(f1classvec[:,i])-1)
		print("Model %s\t%s\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (nummodel,modelstring, mean, h, meanprecision, hprecision, meanf1, hf1),end="")
		for i in range(nClasses):
			print("\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (meanclass[i], hclass[i], meanprecisionclass[i], hprecisionclass[i], meanf1class[i], hf1class[i]),end="")
		print('')
	print('')
	print("########best nearest neighbors model##########")
	print(savemodel)
	print("")


def crossvalidatePCAR(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,n_neighbors=1):
	if n_components == -1 and doPCA == True:
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	print("")
	uniqClasses, labels = np.unique(labels, return_inverse=True)
	if n_neighbors <= 0:
		Kvec = np.arange(start=1,stop=31,step=1,dtype=np.int32)
	else:
		Kvec = [ n_neighbors ]

	modelvec = ""
	savemean = 99999 
	saveh = 0.0
	savemeanr2 = 0.0
	savehr2 = 0.0
	nummodel = 0

	for c in Kvec:
		nummodel += 1
		modelstring = str(c)
		rmsevec = []
		r2vec = []
		for j in range(10):
			ss = KFold(n_splits=5, shuffle=True, random_state=j)
			y_true=[]
			y_pred=[]
			for train_index, test_index in ss.split(matT):
				train=np.copy(matT[train_index])
				test=np.copy(matT[test_index])
				processed=processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
				y_pred = np.append(y_pred,predictNNSimple(processed.train,processed.test,labels[train_index],c,"regression"))
				y_true = np.append(y_true,labels[test_index])
			rmse = math.sqrt(mean_squared_error(y_true, y_pred))
			r2 = r2_score(y_true,y_pred)
			rmsevec = np.append(rmsevec,rmse)
			r2vec = np.append(r2vec,r2)
		mean, se = np.mean(rmsevec), st.sem(rmsevec)
		h = se * scipy.stats.t._ppf((1+0.95)/2., len(rmsevec)-1)
		meanr2, ser2 = np.mean(r2vec), st.sem(r2vec)
		hr2 = ser2 * scipy.stats.t._ppf((1+0.95)/2., len(r2vec)-1)
		if(mean < savemean):
			savemean = mean
			saveh = h
			modelvec = modelstring
			savemeanr2, saveser2 = np.mean(r2vec), st.sem(r2vec)
			savehr2 = saveser2 * scipy.stats.t._ppf((1+0.95)/2., len(r2vec)-1)
		print("k\t",modelstring,"\trmse\t",mean,"\t+/-\t",h,"\tr2\t",meanr2,"\t+/-\t",hr2)
	print('')
	print("########best nearest neighbors model##########")
	print("k = number of nearest neighbors", modelvec,"rmse",savemean,"+/-",saveh,"r2",savemeanr2,"+/-",savehr2)
	print("")

def crossvalidateSVC(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,C=1.0):
	if n_components == -1 and doPCA == True:
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	modeltype = "classification"
	uniqClasses, labels = np.unique(labels, return_inverse=True)
	nClasses=len(uniqClasses)
	print("Classes: ",uniqClasses)
	print("nClasses: ",nClasses)
	print("")
	print("model\tparameters=C\tavg. weighted recall with CI\t avg. weighted precision with CI\t avg. weighted F1-score with CI",end="")
	for i in range(nClasses):
		print("\trecall %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tprecision %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tF1-score %s" % (uniqClasses[i]), end="")
	print("")		    
	if C < 0.0:
		Cvec = np.power(2,np.arange(start=-5,stop=15,step=1,dtype=np.float))
	else:
		Cvec = [ C ]
	modelvec = ""
	savemean = -9999 
	saveh = 0.0
	nummodel = 0
	savemodel = ""
	for C in Cvec:
		modelstring = str(C)
		nummodel += 1
		recallvec = []
		precisionvec = []
		f1vec = []
		recallclassvec = np.array([])
		precisionclassvec = np.array([])
		f1classvec = np.array([])
		meanclass=np.zeros(nClasses)
		meanprecisionclass = np.zeros(nClasses)
		meanf1class = np.zeros(nClasses)
		seclass=np.zeros(nClasses)
		seprecisionclass=np.zeros(nClasses)
		sef1class=np.zeros(nClasses)
		hclass=np.zeros(nClasses)
		hprecisionclass=np.zeros(nClasses)
		hf1class=np.zeros(nClasses)
		for j in range(10):
			ss = KFold(n_splits=5, shuffle=True, random_state=j)
			y_true=[]
			y_pred=[]
			for train_index, test_index in ss.split(matT):
				train=np.copy(matT[train_index])
				test=np.copy(matT[test_index])
				processed=processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
				clf = SVC(kernel='linear', C=C)
				clf.fit(processed.train, labels[train_index])
				y_pred = np.append(y_pred,clf.predict(processed.test))
				y_true = np.append(y_true,labels[test_index])
			recall = recall_score(y_true, y_pred, average='weighted')
			precision = precision_score(y_true, y_pred, average='weighted')
			f1 = f1_score(y_true, y_pred, average='weighted')
			recallvec = np.append(recallvec,recall)
			precisionvec = np.append(precisionvec,precision)
			f1vec = np.append(f1vec,f1)
			recallclass = recall_score(y_true,y_pred,average=None)
			precisionclass = precision_score(y_true,y_pred,average=None)
			f1class =  f1_score(y_true,y_pred,average=None)
			if(j==0):
				recallclassvec = recallclass
				precisionclassvec = precisionclass
				f1classvec = f1class
			else:
				recallclassvec = np.vstack([recallclassvec,recallclass])
				precisionclassvec =  np.vstack([precisionclassvec,precisionclass])
				f1classvec =  np.vstack([f1classvec,f1class])
		mean, se = np.mean(recallvec), st.sem(recallvec)
		meanprecision, seprecision = np.mean(precisionvec), st.sem(precisionvec)
		meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
		h = se * scipy.stats.t._ppf((1+0.95)/2., len(recallvec)-1)
		hprecision = seprecision * scipy.stats.t._ppf((1+0.95)/2., len(precisionvec)-1)
		hf1 = sef1 * scipy.stats.t._ppf((1+0.95)/2., len(f1vec)-1)
		if(meanf1 > savemean):
			savemean=meanf1
			saveh=hf1
			modelvec=modelstring
			savemodel="Model "+str(nummodel)
		for i in range(0,nClasses):
			meanclass[i], seclass[i] = np.mean(recallclassvec[:,i]), st.sem(recallclassvec[:,i])
			meanf1class[i], sef1class[i] = np.mean(f1classvec[:,i]), st.sem(f1classvec[:,i])
			meanprecisionclass[i], seprecisionclass[i] = np.mean(precisionclassvec[:,i]), st.sem(precisionclassvec[:,i])
			hclass[i] =  seclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(recallclassvec[:,i])-1)
			hprecisionclass[i] =  seprecisionclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(precisionclassvec[:,i])-1)
			hf1class[i] =  sef1class[i] * scipy.stats.t._ppf((1+0.95)/2., len(f1classvec[:,i])-1)
		print("Model %s\t%s\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (nummodel,modelstring, mean, h, meanprecision, hprecision, meanf1, hf1),end="")
		for i in range(nClasses):
			print("\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (meanclass[i], hclass[i], meanprecisionclass[i], hprecisionclass[i], meanf1class[i], hf1class[i]),end="")
		print('')
	print('')
	print("########best linear SVM model##########")
	print(savemodel)
	print("")

def crossvalidateSVR(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,C=-1,epsilon=-1):
	modeltype="regression"
	uniqClasses, labels = np.unique(labels, return_inverse=True)
	if C < 0.0:
		Cvec = np.power(2,np.arange(start=-5,stop=15,step=1,dtype=np.float))
	else:
		Cvec = [ C ]
	if epsilon < 0.0:
		EpsVec = [0, 0.01, 0.1, 0.5, 1, 2, 4]
	else:
		EpsVec = [ epsilon ]
	modelvec = ""
	savemean = 99999 
	saveh = 0.0
	savemeanr2 = 0.0
	savehr2 = 0.0
	if n_components == -1 and doPCA == True:
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	for C in Cvec:
		for eps in EpsVec:
			modelstring = str(C)+"-"+str(eps)
			rmsevec = []
			r2vec = []
			for j in range(10):
				ss = KFold(n_splits=5, shuffle=True, random_state=j)
				y_true=[]
				y_pred=[]
				for train_index, test_index in ss.split(matT):
					train = np.copy(matT[train_index])
					test = np.copy(matT[test_index])
					processed=processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
					clf = SVR(kernel='linear', C=C, epsilon=eps)
					clf.fit(processed.train, labels[train_index])
					y_pred = np.append(y_pred,clf.predict(processed.test))
					y_true = np.append(y_true,labels[test_index])
				rmse = math.sqrt(mean_squared_error(y_true, y_pred))
				r2 = r2_score(y_true,y_pred)
				rmsevec = np.append(rmsevec,rmse)
				r2vec = np.append(r2vec,r2)
			mean, se = np.mean(rmsevec), st.sem(rmsevec)
			h = se * scipy.stats.t._ppf((1+0.95)/2., len(rmsevec)-1)
			meanr2, ser2 = np.mean(r2vec), st.sem(r2vec)
			hr2 = ser2 * scipy.stats.t._ppf((1+0.95)/2., len(r2vec)-1)
			if(mean < savemean):
				savemean = mean
				saveh = h
				modelvec = modelstring
				savemeanr2, saveser2 = np.mean(r2vec), st.sem(r2vec)
				savehr2 = saveser2 * scipy.stats.t._ppf((1+0.95)/2., len(r2vec)-1)
			print("C-epsilon\t",modelstring,"\trmse\t",mean,"\t+/-\t",h,"\tr2\t",meanr2,"\t+/-\t",hr2)
	print('')
	print("########best linear SVM model##########")
	print(modelvec,"rmse",savemean,"+/-",saveh,"r2",savemeanr2,"+/-",savehr2)
	print("")

def crossvalidateSVCrbf(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,C=1,gamma=1):

	if C < 0.0:
		Cvec = np.power(2,np.arange(start=-5,stop=15,step=1,dtype=np.float))
	else:
		Cvec = [ C ]
	if gamma < 0.0:
		gvec = np.power(2.0,np.arange(start=-15,stop=3,step=1,dtype=np.float))
	else:
		gvec = [ gamma ]
	modelvec = ""
	savemean = -9999.0 
	saveh = 0.0
	nummodel = 0
	if n_components == -1 and doPCA == True:
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	modeltype = "classification"
	uniqClasses, labels = np.unique(labels, return_inverse=True)
	nClasses=len(uniqClasses)
	print("Classes: ",uniqClasses)
	print("nClasses: ",nClasses)
	print("")
	print("model\tparameters=C:gamma\tavg. weighted recall with CI\t avg. weighted precision with CI\t avg. weighted F1-score with CI",end="")
	for i in range(nClasses):
		print("\trecall %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tprecision %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tF1-score %s" % (uniqClasses[i]), end="")
	print("")
	for C in Cvec:
		for g in gvec:
			modelstring = str(C)+"-"+str(g)
			nummodel += 1
			recallvec = []
			precisionvec = []
			f1vec = []
			recallclassvec = np.array([])
			precisionclassvec = np.array([])
			f1classvec = np.array([])
			meanclass=np.zeros(nClasses)
			meanprecisionclass = np.zeros(nClasses)
			meanf1class = np.zeros(nClasses)
			seclass=np.zeros(nClasses)
			seprecisionclass=np.zeros(nClasses)
			sef1class=np.zeros(nClasses)
			hclass=np.zeros(nClasses)
			hprecisionclass=np.zeros(nClasses)
			hf1class=np.zeros(nClasses)
			for j in range(10):
				ss = KFold(n_splits=5, shuffle=True, random_state=j)
				y_true = []
				y_pred = []
				for train_index, test_index in ss.split(matT):
					train = np.copy(matT[train_index])
					test = np.copy(matT[test_index])
					processed=processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
					clf = SVC(kernel='rbf', C=C, gamma=g)
					clf.fit(processed.train, labels[train_index])
					y_pred = np.append(y_pred,clf.predict(processed.test))
					y_true = np.append(y_true,labels[test_index])
				recall = recall_score(y_true, y_pred, average='weighted')
				precision = precision_score(y_true, y_pred, average='weighted')
				f1 = f1_score(y_true, y_pred, average='weighted')
				recallvec = np.append(recallvec,recall)
				precisionvec = np.append(precisionvec,precision)
				f1vec = np.append(f1vec,f1)
				recallclass = recall_score(y_true,y_pred,average=None)
				precisionclass = precision_score(y_true,y_pred,average=None)
				f1class =  f1_score(y_true,y_pred,average=None)
				if(j==0):
					recallclassvec = recallclass
					precisionclassvec = precisionclass
					f1classvec = f1class
				else:
					recallclassvec = np.vstack([recallclassvec,recallclass])
					precisionclassvec =  np.vstack([precisionclassvec,precisionclass])
					f1classvec =  np.vstack([f1classvec,f1class])
			mean, se = np.mean(recallvec), st.sem(recallvec)
			meanprecision, seprecision = np.mean(precisionvec), st.sem(precisionvec)
			meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
			h = se * scipy.stats.t._ppf((1+0.95)/2., len(recallvec)-1)
			hprecision = seprecision * scipy.stats.t._ppf((1+0.95)/2., len(precisionvec)-1)
			hf1 = sef1 * scipy.stats.t._ppf((1+0.95)/2., len(f1vec)-1)
			if(meanf1 > savemean):
				savemean=meanf1
				saveh=hf1
				modelvec=modelstring
				savemodel="Model "+str(nummodel)
			for i in range(0,nClasses):
				meanclass[i], seclass[i] = np.mean(recallclassvec[:,i]), st.sem(recallclassvec[:,i])
				meanf1class[i], sef1class[i] = np.mean(f1classvec[:,i]), st.sem(f1classvec[:,i])
				meanprecisionclass[i], seprecisionclass[i] = np.mean(precisionclassvec[:,i]), st.sem(precisionclassvec[:,i])
				hclass[i] =  seclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(recallclassvec[:,i])-1)
				hprecisionclass[i] =  seprecisionclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(precisionclassvec[:,i])-1)
				hf1class[i] =  sef1class[i] * scipy.stats.t._ppf((1+0.95)/2., len(f1classvec[:,i])-1)
			print("Model %s\t%s\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (nummodel,modelstring, mean, h, meanprecision, hprecision, meanf1, hf1),end="")
			for i in range(nClasses):
				print("\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (meanclass[i], hclass[i], meanprecisionclass[i], hprecisionclass[i], meanf1class[i], hf1class[i]),end="")
			print('')

	print("########best RBF SVM model##########")
	print(modelvec,"\t",savemean,"\t",saveh)
	print("")


def advancedGTC(train,labels,test,n_neighbors=1,representation="modes",niter=200,k=0,m=0,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,predict_mode="bayes",prior="equiprobable",regularization=-1.0,rbf_width_factor=-1.0):
	if k<=0:
		k=int(math.sqrt(5*math.sqrt(train.shape[0])))+2
	if m<=0:
		m=int(math.sqrt(k))
	if n_components==-1 and doPCA:
		pca=PCA(random_state=random_state)
		pca.fit(matT)
		n_components=np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance in whole data set = %s\n" % n_components)
	if regularization < 0.0 :
		l = 0.1
	else:
		l = regularization
	if rbf_width_factor <= 0.0:
		s = 0.3 
	else:
		s = rbf_width_factor 
	processed = processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
	initialModel = initialize(processed.train,k,m,s,random_state=random_state)
	optimizedModel = optimize(processed.train,initialModel,l,niter,0)
	prediction = advancedPredictBayes(initialModel,optimizedModel,labels,processed.test,prior)
	return prediction

def advancedPredictBayes(initialModel,optimizedModel,labels,newT,prior="equiprobable"):
	predicted = {}
	cl = classMap(optimizedModel,labels,prior)
	activityModel=cl.nodeClassP
	projected = projection(initialModel,optimizedModel,newT)
	predicted["optimizedModel"]=optimizedModel
	predicted["initialModel"]=initialModel
	predicted["indiv_projections"]=projected
	predicted["indiv_probabilities"] = np.matmul(projected.matR,activityModel)
	predicted["indiv_predictions"] = np.argmax(predicted["indiv_probabilities"],axis=1)
	predicted["group_projections"] =  np.mean(projected.matR,axis=0)
	predicted["group_probabilities"] = np.matmul(predicted["group_projections"],activityModel)
	predicted["uniqClasses"]=cl.uniqClasses	
	return predicted

def GTC(train,labels,test,k,m,s,l,n_neighbors=1,niter=200,representation="modes",doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,predict_mode="bayes",prior="equiprobable"):
	processed = processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
	initialModel = initialize(processed.train,k,m,s,random_state=random_state)
	optimizedModel = optimize(processed.train,initialModel,l,niter,0)
	if predict_mode=="knn":
		prediction = predictNN(initialModel,optimizedModel,labels,processed.test,"classification",n_neighbors,representation,prior)
	elif predict_mode=="bayes":
		prediction = predictBayes(initialModel,optimizedModel,labels,processed.test,prior)
	return prediction

def GTR(train,labels,test,k,m,s,l,n_neighbors=1,niter=200,representation="modes",doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	processed = processTrainTest(train,test,doPCA,n_components)
	initialModel = initialize(processed.train,k,m,s,random_state=random_state)
	optimizedModel = optimize(processed.train,initialModel,l,niter,0)
	prediction = predictNN(initialModel=initialModel,optimizedModel=optimizedModel,labels=labels,newT=processed.test,modeltype="regression",n_neighbors=n_neighbors,representation=representation)
	return prediction


def crossvalidateGTC(matT,labels,n_neighbors=1,representation="modes",niter=200,k=0,m=0,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,predict_mode="bayes",prior="equiprobable",regularization=-1.0,rbf_width_factor=-1.0):
	print('k = sqrt(grid size), m = sqrt(radial basis function grid size), l = regularization, s = RBF width factor')
	uniqClasses, labels = np.unique(labels, return_inverse=True)
	nClasses=len(uniqClasses)
	print("Classes: ",uniqClasses)
	print("nClasses: ",nClasses)
	print("")
	print("model\tparameters=k:m:s:l\tavg. weighted recall with CI\t avg. weighted precision with CI\t avg. weighted F1-score with CI",end="")
	for i in range(nClasses):
		print("\trecall %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tprecision %s" % (uniqClasses[i]), end="")
	for i in range(nClasses):
		print("\tF1-score %s" % (uniqClasses[i]), end="")
	print("")
	if k==0:
		k=int(math.sqrt(5*math.sqrt(matT.shape[0])))+2
	if m==0:
		m=int(math.sqrt(k))
	if n_components==-1 and doPCA:
		pca=PCA(random_state=random_state)
		pca.fit(matT)
		n_components=np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance in whole data set = %s\n" % n_components)
	if regularization < 0.0 :
		lvec = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 ]
	else:
		lvec = [ regularization ]
	if rbf_width_factor < 0.0:
		svec = [ 0.25, 0.5, 1.0, 1.50, 2.0 ]
	else:
		svec = [ rbf_width_factor ]
	saveh=0.0
	savemean=-9999
	modelvec=""
	nummodel = 0
	savemodel = ""
	for s in svec:
		for l in lvec:
			modelstring=str(k)+':'+str(m)+":"+str(s)+":"+str(l)
			nummodel+=1
			recallvec = []
			precisionvec = []
			f1vec = []
			recallclassvec = np.array([])
			precisionclassvec = np.array([])
			f1classvec = np.array([])
			meanclass=np.zeros(nClasses)
			meanprecisionclass = np.zeros(nClasses)
			meanf1class = np.zeros(nClasses)
			seclass=np.zeros(nClasses)
			seprecisionclass=np.zeros(nClasses)
			sef1class=np.zeros(nClasses)
			hclass=np.zeros(nClasses)
			hprecisionclass=np.zeros(nClasses)
			hf1class=np.zeros(nClasses)
			for j in range(10):
				ss = KFold(n_splits=5, shuffle=True, random_state=j)
				y_true=[]
				y_pred=[]
				for train_index, test_index in ss.split(matT):
					train=np.copy(matT[train_index])
					test=np.copy(matT[test_index])
					prediction=GTC(train=train,labels=labels[train_index],test=test,k=k,m=m,s=s,l=l,n_neighbors=n_neighbors,niter=niter,representation=representation,doPCA=doPCA,n_components=n_components,random_state=random_state,missing=missing,missing_strategy=missing_strategy,predict_mode=predict_mode,prior=prior)
					y_true=np.append(y_true,labels[test_index])
					y_pred=np.append(y_pred,prediction)
				recall = recall_score(y_true, y_pred, average='weighted')
				precision = precision_score(y_true, y_pred, average='weighted')
				f1 = f1_score(y_true, y_pred, average='weighted')
				recallvec = np.append(recallvec,recall)
				precisionvec = np.append(precisionvec,precision)
				f1vec = np.append(f1vec,f1)
				recallclass = recall_score(y_true,y_pred,average=None)
				precisionclass = precision_score(y_true,y_pred,average=None)
				f1class =  f1_score(y_true,y_pred,average=None)
				if(j==0):
					recallclassvec = recallclass
					precisionclassvec = precisionclass
					f1classvec = f1class
				else:
					recallclassvec = np.vstack([recallclassvec,recallclass])
					precisionclassvec =  np.vstack([precisionclassvec,precisionclass])
					f1classvec =  np.vstack([f1classvec,f1class])
			mean, se = np.mean(recallvec), st.sem(recallvec)
			meanprecision, seprecision = np.mean(precisionvec), st.sem(precisionvec)
			meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
			h = se * scipy.stats.t._ppf((1+0.95)/2., len(recallvec)-1)
			hprecision = seprecision * scipy.stats.t._ppf((1+0.95)/2., len(precisionvec)-1)
			hf1 = sef1 * scipy.stats.t._ppf((1+0.95)/2., len(f1vec)-1)
			if(meanf1 > savemean):
				savemean=meanf1
				saveh=hf1
				modelvec=modelstring
				savemodel="Model "+str(nummodel)
			for i in range(0,nClasses):
				meanclass[i], seclass[i] = np.mean(recallclassvec[:,i]), st.sem(recallclassvec[:,i])
				meanf1class[i], sef1class[i] = np.mean(f1classvec[:,i]), st.sem(f1classvec[:,i])
				meanprecisionclass[i], seprecisionclass[i] = np.mean(precisionclassvec[:,i]), st.sem(precisionclassvec[:,i])
				hclass[i] =  seclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(recallclassvec[:,i])-1)
				hprecisionclass[i] =  seprecisionclass[i] * scipy.stats.t._ppf((1+0.95)/2., len(precisionclassvec[:,i])-1)
				hf1class[i] =  sef1class[i] * scipy.stats.t._ppf((1+0.95)/2., len(f1classvec[:,i])-1)
			print("Model %s\t%s\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (nummodel,modelstring, mean, h, meanprecision, hprecision, meanf1, hf1),end="")
			for i in range(nClasses):
				print("\t%.2f +/- %.3f\t%.2f +/- %.3f\t%.2f +/- %.3f" % (meanclass[i], hclass[i], meanprecisionclass[i], hprecisionclass[i], meanf1class[i], hf1class[i]),end="")
			print('')

	print('')
	print("########best GTC model##########")
	print(savemodel)	
	print("")
				#classreport= classification_report(labels[test_index],prediction, digits=2)
				#print(classreport)

def crossvalidateGTR(matT,labels,n_neighbors=1,representation="modes",niter=200,k=0,m=0,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234,regularization=-1,rbf_width_factor=-1):
	print('k = sqrt(grid size), m = sqrt(radial basis function grid size), l = regularization, s = RBF width factor')
	if k == 0:
		k = int(math.sqrt(5*math.sqrt(matT.shape[0])))+2
	if m == 0:
		m = int(math.sqrt(k))
	if n_components == -1 and doPCA == True:
		pca=PCA(random_state=random_state)
		pca.fit(matT)
		n_components=np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	if regularization<0.0:
		lvec = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 ]
	else:
		lvec = [ regularization ]
	if rbf_width_factor<0.0:
		svec = [ 0.25, 0.5, 1.0, 1.50, 2.0 ]
	else:
		svec = [ rbf_width_factor ]
	savemean = 999999999
	saveh = 0.0
	modelvec = ""
	savemeanr2 = 0.0
	savehr2 = 0.0
	for s in svec:
		for l in lvec:
			modelstring=str(s)+":"+str(l)
			rmsevec=[]
			r2vec=[]
			for j in range(10):
				ss = KFold(n_splits=5, shuffle=True, random_state=j)
				y_true = []
				y_pred = []
				for train_index, test_index in ss.split(matT):
					train = np.copy(matT[train_index])
					test = np.copy(matT[test_index])
					prediction = GTR(train=train,labels=labels[train_index],test=test,k=k,m=m,s=s,l=l,n_neighbors=n_neighbors,niter=niter,representation=representation,doPCA=doPCA,n_components=n_components,random_state=random_state,missing=missing,missing_strategy=missing_strategy)
					y_pred = np.append(y_pred,prediction)
					y_true = np.append(y_true,labels[test_index])
				rmse = math.sqrt(mean_squared_error(y_true, y_pred))
				r2 = r2_score(y_true,y_pred)
				rmsevec = np.append(rmsevec,rmse)
				r2vec = np.append(r2vec,r2)
			mean, se = np.mean(rmsevec), st.sem(rmsevec)
			h = se * scipy.stats.t._ppf((1.0+0.95)/2., len(rmsevec)-1)
			meanr2, ser2 = np.mean(r2vec), st.sem(r2vec)
			hr2 = ser2 * scipy.stats.t._ppf((1.0+0.95)/2., len(r2vec)-1)
			if(mean < savemean):
				savemean = mean
				saveh = h
				modelvec = modelstring
				savemeanr2, saveser2 = np.mean(r2vec), st.sem(r2vec)
				savehr2 = saveser2 * scipy.stats.t._ppf((1+0.95)/2., len(r2vec)-1)
			print("k:m:s:l\t",str(k)+':'+str(m)+':'+modelstring,"\trmse\t",mean,"\t+/-\t",h,"\tR2\t",meanr2,"\t+/-\t",hr2)
	print('')
	print("########best GTR model##########")
	print(modelvec,"rmse",savemean,"+/-",saveh,"r2",savemeanr2,"+/-",savehr2)
	print("")


def pcaPreprocess(matT,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	if missing:
		imp = Imputer(strategy=missing_strategy, axis=0)
		matT = imp.fit_transform(matT)
	scaler =preprocessing.StandardScaler()
	matT = scaler.fit_transform(matT)
	if n_components == -1:
	    n_components = 0.80
	if doPCA:
		pca = PCA(random_state=random_state,n_components=n_components)
		matT = pca.fit_transform(matT)
		n_components=pca.n_components_
		print("Using %s components explaining %s%% of the variance\n" % (n_components,pca.explained_variance_ratio_.cumsum()[n_components-1]*100))
	return(matT)

def chooseKernel(matT,kerneltype='euclidean'):
	if kerneltype == 'euclidean':
		K = np.divide(1,(1+sklearn.metrics.pairwise_distances(matT, metric="euclidean")))
	elif kerneltype == 'cosine':
		K = (sklearn.metrics.pairwise.cosine_kernel(matT))
	elif kerneltype == 'laplacian':
		K = (sklearn.metrics.pairwise.laplacian_kernel(matT))
	elif kerneltype == 'linear':
		K = (sklearn.metrics.pairwise.linear_kernel(matT))
	elif kerneltype == 'polynomial_kernel':
		K = (sklearn.metrics.pairwise.polynomial_kernel(matT))
	elif kerneltype == 'jaccard':
		K = 1-scipy.spatial.distance.cdist(matT,matT,metric='jaccard')
	scaler = KernelCenterer().fit(K)
	return(scaler.transform(K))


def processTrainTest(train,test,doPCA,n_components,missing=False,missing_strategy='most_frequent',random_state=1234):
	if missing:
		imp = Imputer(strategy=missing_strategy, axis=0)
		train = imp.fit_transform(train)
		test = imp.transform(test)
	scaler = preprocessing.StandardScaler()
	train = scaler.fit_transform(train)
	test = scaler.transform(test)
	if(n_components==-1):
	    n_components=0.80
	if doPCA:
		pca = PCA(random_state=random_state,n_components=n_components)
		train = pca.fit_transform(train)
		test = pca.transform(test)
	return(ReturnProcessedTrainTest(train,test))


def whichExperiment(matT,label,args,useDiscrete=0):
	if  useDiscrete==1 and args.model=='GTM':
		decide = 'crossvalidateGTC'
	elif useDiscrete==0 and args.model=='GTM':
		decide = 'crossvalidateGTR'
	elif useDiscrete==1 and args.model=='SVM':
		decide = 'crossvalidateSVC'
	elif useDiscrete==0 and args.model=='SVM':
		decide = 'crossvalidateSVR'
	elif useDiscrete==1 and args.model=='SVMrbf':
		decide = 'crossvalidateSVCrbf'
	elif useDiscrete==1 and args.model=='PCA':
		decide = 'crossvalidatePCAC'
	elif useDiscrete==0 and args.model=='PCA':
		decide = 'crossvalidatePCAR'
	elif useDiscrete==1 and args.model=='compare':
		decide = 'comparecrossvalidateC'
	elif useDiscrete==0 and args.model=='compare':
		decide = 'comparecrossvalidateR'
	else:
		decide = ''
		exit
	if decide == 'crossvalidateGTC':
		crossvalidateGTC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,n_neighbors=args.n_neighbors,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size,predict_mode=args.predict_mode,prior=args.prior,regularization=args.regularization,rbf_width_factor=args.rbf_width_factor)
	elif decide == 'crossvalidateGTR':
		crossvalidateGTR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,n_neighbors=args.n_neighbors,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size,regularization=args.regularization,rbf_width_factor=args.rbf_width_factor)
	elif decide == 'crossvalidateSVC':
		crossvalidateSVC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,C=args.svm_margin)
	elif decide == 'crossvalidateSVCrbf':
		crossvalidateSVCrbf(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,C=args.svm_margin,gamma=args.svm_gamma)
	elif decide =='crossvalidateSVR':
		crossvalidateSVR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,C=args.svm_margin,epsilon=args.svm_epsilon)
	elif decide =='crossvalidatePCAC':
		crossvalidatePCAC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,n_neighbors=args.n_neighbors)
	elif decide =='crossvalidatePCAR':
		crossvalidatePCAR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,n_neighbors=args.n_neighbors)
	elif decide =='comparecrossvalidateC':
		crossvalidateSVC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,C=args.svm_margin)
		crossvalidateGTC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size,predict_mode=args.predict_mode,prior=args.prior,regularization=args.regularization,rbf_width_factor=args.rbf_width_factor)
	elif decide == 'comparecrossvalidateR':
		crossvalidateSVR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,C=args.svm_margin,epsilon=args.svm_epsilon)
		crossvalidateGTR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,n_neighbors=args.n_neighbors,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size,regularization=args.regularization,rbf_width_factor=args.rbf_width_factor)
	else:
		print("Could not determine which experiment to conduct.")

def printClassPredictions(prediction,output):
	string = ""
	for i in range(len(prediction["uniqClasses"])-1):
		string += str(prediction["uniqClasses"][i])+","
	string +=  str(prediction["uniqClasses"][len(prediction["uniqClasses"])-1])
	np.savetxt(fname=output+"_indiv_probabilities.csv",X=prediction["indiv_probabilities"],delimiter=",",header=string,fmt='%.2f')
	np.savetxt(fname=output+"_indiv_predictions.csv",X=prediction["indiv_predictions"],delimiter=",",header=string,fmt='%.2f')
	np.savetxt(fname=output+"_group_probabilities.csv",X=prediction["group_probabilities"].reshape(1, prediction["group_probabilities"].shape[0]),delimiter=",",header=string,fmt='%.2f')
	print("Wrote to disk:")
	print("%s: individual probabilities" % (output+"_indiv_probabilities.csv"))
	print("%s: individual predictions" % (output+"_indiv_predictions.csv"))
	print("%s: group probabilities" % (output+"_group_probabilities.csv"))
	print("")






def plotMultiPanelGTM(initialModel,optimizedModel,label,output,useDiscrete):
	fig = plt.figure(figsize=(10,10))
	means=optimizedModel.matMeans
	modes=optimizedModel.matModes
	#plot1: GTM means visualization
	ax = fig.add_subplot(221); 
	ax.scatter(means[:, 0], means[:, 1], c=label, cmap=plt.cm.Spectral); plt.axis('tight'); plt.xticks([]), plt.yticks([]); plt.title('Means');
	#plot2: GTM modes visualization
	ax2 = fig.add_subplot(222);
	ax2.scatter(modes[:, 0], modes[:, 1], c=label, cmap=plt.cm.Spectral); plt.axis('tight'); plt.xticks([]), plt.yticks([]); plt.title('Modes');
	#plot3: GTM landscape visualization
	ax3 = fig.add_subplot(223);
	#if it's label data, the landscape is a class map; otherwise, it is a continuous landscape
	if useDiscrete:
		plotClassMap(initialModel,optimizedModel,label)
	else:
		plotLandscape(initialModel,optimizedModel,label)
	#add mapping from mean positions to modes (GTM nodes where the data points have max probability)
	for i in range(label.shape[0]):
		plt.plot([means[i,0],modes[i,0]],[means[i,1],modes[i,1]],color='grey',linewidth=0.5)
		
	#plot4: GTM landscape visualization without means/modes mappings
	ax4 = fig.add_subplot(224);
	if useDiscrete:
		plotClassMapNoPoints(initialModel,optimizedModel,label)
	else:
		plotLandscapeNoPoints(initialModel,optimizedModel,label)
	fig.set_size_inches(16, 13)
	fig.savefig(output+".pdf",format='pdf', dpi=500)
	plt.close(fig)
	print("\nWrote pdf to disk: %s\n" % (output+".pdf"))
