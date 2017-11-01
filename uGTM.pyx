import pandas as pd
import numpy as np
import os
import scipy
import sys
import math
import sklearn
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

#create manifold
def createYMatrixInit(matT,matW,matPhiMPlusOne):
	shap1=matW.shape[0];
	shap2=matPhiMPlusOne.shape[0];
	TheMeans=matT.mean(0)
	DMmeanMatrix = np.zeros([matW.shape[0], matPhiMPlusOne.shape[0]])
	for i in range(shap1):
		for j in range(shap2):
			DMmeanMatrix[i,j] = TheMeans[i]
	MatY = np.dot(matW, np.transpose(matPhiMPlusOne))
	MatY = MatY + DMmeanMatrix
	return(MatY)
   
def createPhiMatrix(matX,matM,numX, numM,sigma):
	Result=np.zeros([numX, numM + 1])
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
	mins=np.zeros([numM, 1])
	maxs=np.zeros([numM, 1])
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


def createFeatureMatrixNIPALS(matT, nMolecules, nDimensions):
	Result = np.zeros([nDimensions, 2])
	threshold = 0.0001
	id = 0
#shortest way: take first column
	t = matT[:,id]
	if (np.sum(t)==0):
		for i in range(nMolecules):
			t[i] = abs(np.random.uniform(-1, 1, size=1))
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
	NormX = sklearn.preprocessing.scale(matX,axis=0, with_mean=True, with_std=True) 
	myProd = np.dot(matU,np.transpose(NormX))
	tinv = np.linalg.solve(matPhiMPlusOne.T.dot(matPhiMPlusOne), matPhiMPlusOne.T)
	Result = np.dot(myProd,np.transpose(tinv))
	return(Result)


def createYMatrix(matW,matPhiMPlusOne):
	Result = np.dot(matW, np.transpose(matPhiMPlusOne))
	return(Result)


class ReturnInitial(object):
	def __init__(self, matX, matM, nSamples, nCenters, rbfWidth, matPhiMPlusOne, matU, matW, matY,betaInv):
		self.matX = matX 
		self.matM = matM 
		self.nCenters = nCenters
		self.nSamples = nSamples
		self.rbfWidth = rbfWidth
		self.matPhiMPlusOne = matPhiMPlusOne
		self.matU = matU
		self.matW = matW
		self.matY = matY
		self.betaInv = betaInv

class ReturnOptimized(object):
	def __init__(self, matW, matY, matP, matR, betaInv, matMeans):
		self.matW = matW
		self.matY = matY
		self.matP = matP
		self.matR = matR
		self.betaInv = betaInv
		self.matMeans = matMeans

class ReturnU(object):
	def __init__(self, matU, betaInv):
		self.matU = matU 
		self.betaInv = betaInv 

def initiliaze(matT,k,m,s,l):
	#create X matrix

	nMolecules = matT.shape[0]
	nDimensions = matT.shape[1]

	nSamples=k*k
	nCenters=m*m

	x = np.linspace(-1, 1, k)
	matX=np.transpose(np.meshgrid(x,x)).reshape(k*k,2)

	#create M matrix
	x = np.linspace(-1, 1, m)
	matM=np.transpose(np.meshgrid(x,x)).reshape(m*m,2)
	#matMnorm=sklearn.preprocessing.normalize(matM,axis=0)

	##################
	#compute width
	rbfWidth=computeWidth(matM,nCenters,s) 

	####################
	#create Phi matrix
	matPhiMPlusOne = createPhiMatrix(matX,matM,nSamples,nCenters,rbfWidth)

	#####################
	#create U matrix: use NIPALS approach since it's somewhat faster
	#start = time.time()
	Uobj = createFeatureMatrixNIPALS(matT, nMolecules, nDimensions)

	#end = time.time(); elapsed = end - start
	#alternative for creating U loading matrix: instead of NIPALS, use PCA (it's slower.....):
#	pca = PCA(n_components=3)
#	pca.fit(matT)
#	matU=(pca.components_.T * np.sqrt(pca.explained_variance_))[:,0:2]
#	betaInv=pca.explained_variance_[2]

	#create W matrix
	matW = createWMatrix(matX,matPhiMPlusOne,Uobj.matU,nDimensions,nCenters)

	#create Y matrix
	matY = createYMatrixInit(matT,matW,matPhiMPlusOne)

	betaInv = Uobj.betaInv
	betaInv = evalBetaInv(matY,Uobj.betaInv) 

	return ReturnInitial(matX, matM, nSamples, nCenters, rbfWidth, matPhiMPlusOne, Uobj.matU, matW, matY, betaInv)


def createDistanceMatrix(matY, matT):
	Result = scipy.spatial.distance.cdist(matY.T,matT,metric='sqeuclidean')	
	return(Result)


def createPMatrix(matD,betaInv,nDimensions):
	beta=1/betaInv
	constante = np.power(((beta)/(2*np.pi)),nDimensions/2)
	matP = constante*np.exp(-(beta/2)*matD)
	return(matP)


def createRMatrix(matP):
	sums = np.sum(matP,axis=0)
	matR = matP / sums[None,:]
	return(matR)

def optimize(matT, initialModel, alpha, niter):
	matD = createDistanceMatrix(initialModel.matY, matT)
	matY = initialModel.matY
	betaInv = initialModel.betaInv
	i = 1;
	diff=1000;
	while i<(niter+1) and diff>0.0001:	
		#expectation
		#start = time.time();
		matP = createPMatrix(matD,betaInv,matT.shape[1])
		#end = time.time(); elapsed = end - start; print("P=",elapsed);start = time.time();
		matR = createRMatrix(matP)
		#end = time.time(); elapsed = end - start; print("R=",elapsed);start = time.time();
		#maximization
		matG = createGMatrix(matR)
		#end = time.time(); elapsed = end - start; print("G=",elapsed);start = time.time();
		matW = optimWMatrix(matR, initialModel.matPhiMPlusOne, matG, matT, betaInv, alpha)
		#end = time.time(); elapsed = end - start; print("W=",elapsed);start = time.time();
		matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
		#end = time.time(); elapsed = end - start; print("Y=",elapsed);start = time.time();
		matD = createDistanceMatrix(matY, matT)
		#end = time.time(); elapsed = end - start; print("D=",elapsed);start = time.time();
		betaInv = optimBetaInv(matR,matD,matT.shape[1])
		#end = time.time(); elapsed = end - start; print("b=",elapsed);start = time.time();
		#objective function
		if i == 1:
			loglike = computelogLikelihood(matP,betaInv,matT.shape[1]);
		else:
			loglikebefore=loglike
			loglike = computelogLikelihood(matP,betaInv,matT.shape[1])
			diff=abs(loglikebefore-loglike)
		#end = time.time(); elapsed = end - start; print("l=",elapsed);
		print("Iter ", i, " ErrorFunction (should go down): ", loglike)
		i += 1
	matMeans = meanPoint(matR, initialModel.matX)
	return ReturnOptimized(matW, matY, matP, matR, betaInv, matMeans)

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

def optimBetaInv(matR,matD,nDimensions):
	nMolecules = matR.shape[1]
	betaInv = np.sum(np.multiply(matR,matD))/(nMolecules*nDimensions)
	return(betaInv)

def meanPoint(matR,matX):
	matMeans = np.dot(np.transpose(matR), matX)
	return(matMeans)


def computelogLikelihood(matP,betaInv,nDimensions):
	nSamples = matP.shape[0]
	nMolecules = matP.shape[1]
	Ptwb = 0.0
	LogLikelihood = 0.0
	prior = 1/nSamples
	LogLikelihood = np.sum(np.log(np.maximum(np.sum(matP,axis=0)*prior,sys.float_info.epsilon)))
	LogLikelihood /= nMolecules
	return(-LogLikelihood)


def evalBetaInv(matY,betaInv):
	Distances = sklearn.metrics.pairwise.euclidean_distances(np.transpose(matY))
	myMin = np.mean(Distances)/2;
	myMin *= myMin;
	if((myMin<betaInv) or (betaInv==0)):
		betaInv = myMin
	if betaInv==0.0:
		print('bad initialization (0 variance), setting variance to random number...');
		betaInv = abs(np.random.uniform(-1, 1, size=1));
	return(betaInv);

