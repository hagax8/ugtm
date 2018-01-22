import pandas as pd
import numpy as np
import os
import scipy
import sys
import math
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KernelCenterer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold 
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import time
import random
from scipy.interpolate import Rbf
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as st
from sklearn.feature_selection import VarianceThreshold

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
	def __init__(self, nodeClassP, activityModel):
		self.nodeClassP = nodeClassP
		self.activityModel = activityModel 

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
	while (i<(niter+1)) and (converged<2):
		#expectation
		#start = time.time();
		matP = createPMatrix(matD,betaInv,initialModel.nDimensions)
		#end = time.time(); elapsed = end - start; print("P=",elapsed);start = time.time();
		matR = createRMatrix(matP)
		#end = time.time(); elapsed = end - start; print("R=",elapsed);start = time.time();
		#maximization
		matG = createGMatrix(matR)
		#end = time.time(); elapsed = end - start; print("G=",elapsed);start = time.time();
		matW = optimLMatrix(matR, initialModel.matPhiMPlusOne, matG, betaInv, alpha)
		#end = time.time(); elapsed = end - start; print("W=",elapsed);start = time.time();
		matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
		#end = time.time(); elapsed = end - start; print("Y=",elapsed);start = time.time();
		matD = KERNELcreateDistanceMatrix(matT, matW, initialModel.matPhiMPlusOne)
		#end = time.time(); elapsed = end - start; print("D=",elapsed);start = time.time();
		betaInv = optimBetaInv(matR,matD,initialModel.nDimensions)
		#end = time.time(); elapsed = end - start; print("b=",betaInv,elapsed);start = time.time();
		#objective function
		if i == 1:
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions);
		else:
			loglikebefore = loglike
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions)
			diff = abs(loglikebefore-loglike)
		#end = time.time(); elapsed = end - start; print("l=",elapsed);
		if verbose == True:
			#print("Iter ", i, " Err: ", loglike," Beta: ",1/betaInv)
                        print("Iter ", i, " Err: ", loglike)
		if diff <= 0.0001:
			converged+=1
		i += 1
	if verbose == True:
		if converged >= 2: 
			print("Converged: ",loglike)
	matY = createYMatrix(matW,initialModel.matPhiMPlusOne)
	matMeans = meanPoint(matR, initialModel.matX)
	matModes = modePoint(matR, initialModel.matX)
	return ReturnOptimized(matW, matY, matP.T, matR.T, betaInv, matMeans, matModes)


def landscape(optimizedModel,activity):
	sums = np.sum(optimizedModel.matR,axis=0)
	landscape = np.dot(activity,optimizedModel.matR) / sums[None,:]
	return np.asarray(landscape)[0,:]

def classMap(optimizedModel,activity):
	uniqClasses, classVector = np.unique(activity, return_inverse=True)
	nClasses = uniqClasses.shape[0]
	nMolecules = optimizedModel.matR.shape[0]
	nSamples = optimizedModel.matR.shape[1]
	nodeClassP = np.zeros([nSamples,nClasses])
	sumClass = np.zeros([nClasses])
	for i in range(nClasses):
		sumClass[i]=(classVector==i).sum()
		for k in range(nSamples):
			nodeClassP[k,i] = optimizedModel.matR[classVector==i,k].sum()/sumClass[i]
	nodeClass = np.argmax(nodeClassP, axis=1)
	return(ReturnClassMap(nodeClassP,nodeClass))


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


def createPMatrix(matD,betaInv,nDimensions):
	matP = np.array([],dtype=np.longdouble) 
	beta = 1/betaInv
	if nDimensions > 100:
		dim = 100
	else:
		dim = nDimensions
	constante = np.power(((beta)/(2*np.pi)),dim/2)
	#constante = 1
	matP = constante*np.exp(-(beta/2)*matD)
	#matP[matP < np.finfo(float).tiny ] = np.finfo(float).tiny
        #threshold_indices = matP == 0 
	#matP[threshold_indices] = sys.float_info.min 
	return(matP)


def createRMatrix(matP):
	matR = np.array([],dtype=np.longdouble)
	sums = np.sum(matP,axis=0)
	ids = np.where(sums == 0)[0]
	if(len(ids)>0):
		print("These guys have 0 probability and can't be mapped:")
		print(ids+1)
	#nk = len(matP[:,1])
	#ratio = 1/nk
	#if len(ids)>0 and ('matR' in locals() or 'matR' in globals()):
	#saved = np.copy(matR)
		#nk = len(matR[:,1])
		#ratio = 1/nk
	matR = matP / sums[None,:]
	#matR=np.delete(matR,ids,axis=1)
	#for i in ids:
	#	matR[:,i] = ratio 
	return(matR)
	#return(ReturnExtendedR(matR,ids))

def optimize(matT, initialModel, alpha, niter, verbose=True):
	matD = createDistanceMatrix(initialModel.matY, matT)
	matY = initialModel.matY
	betaInv = initialModel.betaInv
	i = 1
	diff = 1000
	converged = 0
	while i<(niter+1) and (converged<2):	
		#expectation
		#start = time.time();
		matP = createPMatrix(matD,betaInv,initialModel.nDimensions)
		#end = time.time(); elapsed = end - start; print("P=",elapsed);start = time.time();
		matR=createRMatrix(matP)
	#	matR = obj.matR
	#	ids = obj.ids
#		matT=np.delete(matT,ids,axis=0)
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
		betaInv = optimBetaInv(matR,matD,initialModel.nDimensions)
		#end = time.time(); elapsed = end - start; print("b=",elapsed);start = time.time();
		#objective function
		if i == 1:
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions);
		else:
			loglikebefore=loglike
			loglike = computelogLikelihood(matP,betaInv,initialModel.nDimensions)
			diff=abs(loglikebefore-loglike)
		if diff <= 0.0001:
			converged += 1
		#end = time.time(); elapsed = end - start; print("l=",elapsed);
		if verbose:
			#print("Iter ", i, " Err: ", loglike," Beta: ",1/betaInv)
                        print("Iter ", i, " Err: ", loglike)
		i += 1
	#final iteration to make sure matR fits matD
	if verbose == 1:
		if converged >= 2:
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
	prior = 1/nSamples
	LogLikelihood = np.sum(np.log(np.maximum(np.sum(matP,axis=0)*prior,np.finfo(float).tiny)))
	#LogLikelihood = np.sum(np.log(np.sum(matP,axis=0)*prior))
	#LogLikelihood = np.sum(np.log(np.sum(matP,axis=0)*prior))
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
	plt.scatter(x, y, 175*(10/k), z, cmap=plt.cm.Spectral,marker="s")
	plt.scatter(optimizedModel.matMeans[:,0], optimizedModel.matMeans[:,1], 20, label, cmap=plt.cm.Spectral,edgecolor='black',marker="o")
	plt.title('Landscape')
	plt.xlim(-1.1, 1.1)
	plt.ylim(-1.1, 1.1)
	plt.colorbar()
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])

def plotClassMap(initialModel,optimizedModel,label):
	k = math.sqrt(initialModel.nSamples);
	x = initialModel.matX[:,0]
	y = initialModel.matX[:,1]
	z = classMap(optimizedModel,label).activityModel
	#ti = np.linspace(-1.0, 1.0, k)
	#XI, YI = np.meshgrid(ti, ti)
	#f = interpolate.NearestNDInterpolator(initialModel.matX,z)
	#ZI=f(XI,YI)
	#plt.pcolor(XI, YI, ZI, cmap=plt.cm.Spectral)
	uniqClasses, label = np.unique(label, return_inverse=True)
	plt.scatter(x, y, 175*(10/k), z, cmap=plt.cm.Spectral,marker="s",alpha=0.3)
	plt.scatter(optimizedModel.matMeans[:,0], optimizedModel.matMeans[:,1], 20, label, cmap=plt.cm.Spectral,edgecolor='black',marker="o")
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

def predictNN(initialModel,optimizedModel,labels,newT,modeltype,n_neighbors=1,representation="modes"):
	if modeltype=='regression':
		activityModel=landscape(optimizedModel,labels)
	elif modeltype=='classification':
		n_neibhbors=1
		activityModel=classMap(optimizedModel,labels).activityModel
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
		#predicted = np.average(activityModel[nnID],axis=1)
		#print(activityModel[nnID])
	else:
		nnID = fitted.kneighbors(rep,return_distance=False)
		predicted=activityModel[nnID]
	return predicted

def optimizeSVC(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	if n_components == -1 and doPCA == True:
#		scaler = preprocessing.StandardScaler().fit(matT)
#		scaler.fit(matT)
#		td = scaler.transform(matT)
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	modeltype = "classification"
	Cvec = np.power(2,np.arange(start=-5,stop=15,step=1,dtype=np.float))
	modelvec = ""
	savemean = 0.0
	saveh = 0.0
	for C in Cvec:
		modelstring = str(C)
		recallvec = []
		for j in range(10):
			ss = KFold(n_splits=5, shuffle=True, random_state=j)
			y_true=[]
			y_pred=[]
			for train_index, test_index in ss.split(matT):
				train=np.copy(matT[train_index])
				test=np.copy(matT[test_index])
				processed=processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
#				scaler = MinMaxScaler(feature_range=(-1, 1))
#				scaler.fit(train)
#				train = scaler.transform(train)
#				test = scaler.transform(test)
#				if doPCA:
##					scaler = preprocessing.StandardScaler().fit(train)
#					train = scaler.transform(train)
#					test = scaler.transform(test)
#					pca = PCA(n_components=n_components)
#					pca.fit(train)
#					train = pca.transform(train)
#					test = pca.transform(test)
				clf = SVC(kernel='linear', C=C)
				clf.fit(processed.train, labels[train_index])
				y_pred = np.append(y_pred,clf.predict(processed.test))
				y_true = np.append(y_true,labels[test_index])
			recall = recall_score(y_true, y_pred, average='weighted')
			recallvec = np.append(recallvec,recall)
		mean, se = np.mean(recallvec), st.sem(recallvec)
		h = se * scipy.stats.t._ppf((1+0.95)/2., len(recallvec)-1)
		if(mean-h > savemean):
			savemean=mean
			saveh=h
			modelvec=modelstring
		print("C",modelstring,"avg. weighted recall",mean,"±",h,sep="\t")
	print("########best linear SVM model##########")
	print(modelvec,savemean,saveh)
	print("")

def optimizeSVR(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	modeltype="regression"
	Cvec = np.power(2,np.arange(start=-5,stop=15,step=1,dtype=np.float))
	EpsVec = [0, 0.01, 0.1, 0.5, 1, 2, 4]
	modelvec = ""
	savemean = 0.0
	saveh = 0.0
	savemeanr2 = 0.0
	savehr2 = 0.0
	if n_components == -1 and doPCA == True:
#		scaler = preprocessing.StandardScaler().fit(matT)
#		scaler.fit(matT)
#		td = scaler.transform(matT)
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
#					scaler = MinMaxScaler(feature_range=(-1, 1))
#					scaler.fit(train)
#					train = scaler.transform(train)
#					test = scaler.transform(test)
#					if doPCA:
#						scaler = preprocessing.StandardScaler().fit(train)
#						train = scaler.transform(train)
#						test = scaler.transform(test)
#						pca = PCA(n_components=n_components)
#						pca.fit(train)
#						train=pca.transform(train)
#						test=pca.transform(test)
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
			if(mean-h > savemean):
				savemean = mean
				saveh = h
				modelvec = modelstring
				savemeanr2, saveser2 = np.mean(r2vec), st.sem(r2vec)
				savehr2 = saveser2 * scipy.stats.t._ppf((1+0.95)/2., len(r2vec)-1)
			print("C-epsilon",modelstring,"rmse",mean,"±",h,"r2",meanr2,"±",hr2,sep="\t")
	print("########best linear SVM model##########")
	print(modelvec,"rmse",savemean,"±",saveh,"r2",savemeanr2,"±",savehr2)
	print("")


def optimizeSVCrbf(matT,labels,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	modeltype="classification"
	Cvec = np.power(2,np.arange(start=-5,stop=15,step=1,dtype=np.float))
	gvec = np.power(2,np.arange(start=-15,stop=3,step=1,dtype=np.float))
	modelvec = ""
	savemean = 0.0
	saveh = 0.0
	if n_components == -1 and doPCA == True:
		#scaler = preprocessing.StandardScaler().fit(matT)
		#dt = scaler.transform(matT)
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components = np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	for C in Cvec:
		for g in gvec:
			modelstring = str(C)+"-"+str(g)
			recallvec = []
			for j in range(10):
				ss = KFold(n_splits=5, shuffle=True, random_state=j)
				y_true = []
				y_pred = []
				for train_index, test_index in ss.split(matT):
					train = np.copy(matT[train_index])
					test = np.copy(matT[test_index])
					processed=processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
#					scaler = MinMaxScaler(feature_range=(-1, 1))
#					scaler.fit(train)
#					train = scaler.transform(train)
#					test = scaler.transform(test)
#					if doPCA:
#						scaler = preprocessing.StandardScaler().fit(train)
#						train = scaler.transform(train)
#						test = scaler.transform(test)
#						pca = PCA(n_components=n_components)
#						pca.fit(train)
#						train = pca.transform(train)
#						test = pca.transform(test)
					clf = SVC(kernel='rbf', C=C, gamma=g)
					clf.fit(processed.train, labels[train_index])
					y_pred = np.append(y_pred,clf.predict(processed.test))
					y_true = np.append(y_true,labels[test_index])
				recall = recall_score(y_true, y_pred, average='weighted')
				recallvec = np.append(recallvec,recall)
			mean, se = np.mean(recallvec), st.sem(recallvec)
			h = se * scipy.stats.t._ppf((1+0.95)/2., len(recallvec)-1)
			if(mean-h > savemean):
				savemean = mean
				saveh = h
				modelvec = modelstring
			print("C-gamma",modelstring,"avg. weighted recall",mean,"±",h,sep="\t")
	print("########best RBF SVM model##########")
	print(modelvec,savemean,saveh,sep="\t")
	print("")

def GTC(train,labels,test,k,m,s,l,n_neighbors=1,niter=200,representation="modes",doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	processed = processTrainTest(train,test,doPCA,n_components,missing,missing_strategy)
	initialModel = initialize(processed.train,k,m,s,random_state=random_state)
	optimizedModel = optimize(processed.train,initialModel,l,niter,0)
	prediction = predictNN(initialModel,optimizedModel,labels,processed.test,"classification",n_neighbors,representation)
	return prediction

def GTR(train,labels,test,k,m,s,l,n_neighbors=1,niter=200,representation="modes",doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	processed = processTrainTest(train,test,doPCA,n_components)
	initialModel = initialize(processed.train,k,m,s,random_state=random_state)
	optimizedModel = optimize(processed.train,initialModel,l,niter,0)
	prediction = predictNN(initialModel,optimizedModel,labels,processed.test,"regression",n_neighbors,representation)
	return prediction

def optimizeGTC(matT,labels,n_neighbors=1,representation="modes",niter=200,k=0,m=0,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	if k==0:
		k=int(math.sqrt(5*math.sqrt(matT.shape[0])))+2
	if m==0:
		m=int(math.sqrt(k))
	if n_components==-1 and doPCA:
		pca=PCA(random_state=random_state)
		pca.fit(matT)
		n_components=np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance in whole data set = %s\n" % n_components)
	svec = [ 0.25, 0.5, 1.0, 1.50, 2.0 ]
	lvec = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 ]
	savemean=0.0
	saveh=0.0
	modelvec=""
	for s in svec:
		for l in lvec:
			modelstring=str(s)+"-"+str(l)
			recallvec=[]
			for j in range(10):
				ss = KFold(n_splits=5, shuffle=True, random_state=j)
				y_true=[]
				y_pred=[]
				for train_index, test_index in ss.split(matT):
					train=np.copy(matT[train_index])
					test=np.copy(matT[test_index])
					prediction=GTC(train,labels[train_index],test,k,m,s,l,n_neighbors,niter,representation,doPCA=doPCA,n_components=n_components,random_state=random_state)
					y_true=np.append(y_true,labels[test_index])
					y_pred=np.append(y_pred,prediction)
				recall=recall_score(y_true, y_pred, average='weighted') 
				recallvec=np.append(recallvec,recall)
			mean, se = np.mean(recallvec), st.sem(recallvec)
			h = se * scipy.stats.t._ppf((1+0.95)/2., len(recallvec)-1)
			if(mean > savemean):
				savemean=mean
				saveh=h
				modelvec=modelstring
			print("s-l",modelstring,"avg. weighted recall",mean,"±",h,sep="\t")
	print("########best GTC model##########")
	print(modelvec,savemean,saveh)	
	print("")
				#classreport= classification_report(labels[test_index],prediction, digits=2)
				#print(classreport)

def optimizeGTR(matT,labels,n_neighbors=1,representation="modes",niter=200,k=0,m=0,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	if k == 0:
		k = int(math.sqrt(5*math.sqrt(matT.shape[0])))+2
	if m == 0:
		m = int(math.sqrt(k))
	if n_components == -1 and doPCA == True:
		pca=PCA(random_state=random_state)
		pca.fit(matT)
		n_components=np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Using number of components explaining 80%% of the variance = %s\n" % n_components)
	svec = [ 0.25, 0.5, 1.0, 1.50, 2.0 ]
	lvec = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 ]
	savemean = 9999999999999 
	saveh = 0.0
	modelvec = ""
	savemeanr2 = 0.0
	savehr2 = 0.0
	for s in svec:
		for l in lvec:
			modelstring=str(s)+"-"+str(l)
			rmsevec=[]
			r2vec=[]
			for j in range(10):
				ss = KFold(n_splits=5, shuffle=True, random_state=j)
				y_true = []
				y_pred = []
				for train_index, test_index in ss.split(matT):
					train = np.copy(matT[train_index])
					test = np.copy(matT[test_index])
					prediction = GTR(train,labels[train_index],test,k,m,s,l,n_neighbors,niter,representation,doPCA,n_components,random_state=random_state)
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
			print("s-l",modelstring,"rmse",mean,"±",h,"R2",meanr2,"±",hr2,sep="\t")
	print("########best GTR model##########")
	print(modelvec,"rmse",savemean,"±",saveh,"r2",savemeanr2,"±",savehr2)
	print("")


def pcaPreprocess(matT,doPCA=False,n_components=-1,missing=False,missing_strategy='most_frequent',random_state=1234):
	if(n_components>100):
		n_components=100
	if missing:
		imp = Imputer(strategy=missing_strategy, axis=0)
		matT = imp.fit_transform(matT)
	sel = VarianceThreshold()
	matT = sel.fit_transform(matT)
#	scaler = MinMaxScaler(feature_range=(-1, 1))
#	matT = scaler.fit_transform(matT)
	if n_components == -1 and doPCA:
		pca = PCA(random_state=random_state)
		pca.fit(matT)
		n_components=np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.8)+1
		print("Number of components explaining 80%% of the variance = %s\n" % n_components)
	if(n_components>100):   
		n_components=100
	if doPCA:
#		scaler = preprocessing.StandardScaler()
#		matT = scaler.fit_transform(matT)
		pca = PCA(random_state=random_state,n_components=n_components)
#PCA on covariance matrix 
		matT = pca.fit_transform(matT)
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
		imp.fit(train)
		train = imp.transform(train)
		test = imp.transform(test)
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler.fit(train)
	train = scaler.transform(train)
	test = scaler.transform(test)
	if n_components==-1 and doPCA==True:
		#scaler = preprocessing.StandardScaler().fit(train)
		#scaler.fit(train)
		#td = scaler.transform(train)
		pca = PCA(random_state=random_state)
		pca.fit(train)
		n_components = np.searchsorted(pca.explained_variance_ratio.cumsum(),0.8)+1
		print("Number of components explaining 80%% of the variance in training set = %s\n" % n_components)
	if doPCA:
	#	scaler = preprocessing.StandardScaler().fit(train)
	#	train = scaler.transform(train)
	#	test = scaler.transform(test)
		pca = PCA(random_state=random_state,n_components=n_components)
		pca.fit(train)
		train = pca.transform(train)
		test = pca.transform(test)
	return(ReturnProcessedTrainTest(train,test))
