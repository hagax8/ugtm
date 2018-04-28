import uGTM
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt, mpld3
import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt
import mpl_toolkits.mplot3d.axes3d as p3
import argparse
import scipy
from scipy.spatial.distance import cdist
import math
import csv


############ argument parsing
parser = argparse.ArgumentParser(description='Generate and assess GTM maps for classification or regression.')
parser.add_argument('--data', help='data file in csv format without header (must be a similarity matrix if --model kGTM is set, otherwise not)', dest='filenamedat')
parser.add_argument('--labels', help='label file in csv format without header', dest='filenamelbls')
parser.add_argument('--ids', help='label file in csv format without header', dest='filenameids')
parser.add_argument('--testids', help='label file in csv format without header', dest='filenametestids')
parser.add_argument('--labeltype', help='labels are continuous or discrete', dest='labeltype', choices=['continuous','discrete'])
parser.add_argument('--usetest', help='use S or swiss roll or iris test data', dest='usetest',choices=['s','swiss','iris'])
parser.add_argument('--model',help='GTM model, kernel GTM model, SVM, PCA or comparison between: GTM, kGTM, LLE and tSNE for simple visualization, GTM and SVM for regression or classification (when --crossvalidate is set); the benchmarked parameters for GTM are regularization and rbf_width_factor for given grid_size and rbf_grid_size', dest='model',choices=['GTM','kGTM','SVM','PCA','t-SNE','compare']) 
parser.add_argument('--output',help='output name', dest='output')
parser.add_argument('--crossvalidate',help='show best l (regularization coefficient) and w combination (RBF width factor) for classification or regression, with default grid size parameter k = sqrt(5*sqrt(Nfeatures))+2) and RBF grid size parameter m = sqrt(k); you can also set the 4 parameters and run only one model width --rbf_width_factor, regularization, grid_size and rbf_grid_size', action='store_true' )
parser.add_argument('--pca', help='do PCA preprocessing; if --n_components is not set, will use number of PCs explaining 80%% of variance', action='store_true')
parser.add_argument('--missing', help='there is missing data (encoded by NA)', action='store_true')
parser.add_argument('--test', help='test data; only available for GTM classification at the moment (define training data with --data and training labels with --labels with --labeltype discrete)', dest='test')
parser.add_argument('--missing_strategy',help='missing data strategy, missing values encoded by NA; default is median', const='median', type=str, default='median', nargs='?',dest='missing_strategy',choices=['mean','median','most_frequent'])
parser.add_argument('--predict_mode',help='predict mode for GTM classification: default is bayes for an equiprobable class prediction, you can change this to knn; knn is the only one available for PCA and t-SNE, this option is only useful for GTM',const='bayes',type=str,default='bayes',nargs='?',dest='predict_mode',choices=['bayes','knn'])
parser.add_argument('--prior',help='type of prior for GTM classification map and prediction model: you can choose equiprobable classes (prior any class=1/nClasses) or to estimate classes from the training set (prior class 1 = sum(class 1 instances in training set)/sum(instances in training set))',const='equiprobable',type=str,default='equiprobable',nargs='?',dest='prior',choices=['equiprobable','estimated'])
parser.add_argument('--n_components',help='set number of components for PCA pre-processing, if --pca flag is present', const=-1, type=int, default=-1, nargs='?',dest='n_components')
parser.add_argument('--percentage_components',help='set number of components for PCA pre-processing, if --pca flag is present', const=0.80, type=float, default=0.80, nargs='?',dest='n_components')
parser.add_argument('--regularization',help='set regularization factor, default: 0.1; set this to -1 to crossvalidate when using --crossvalidate', type=float, dest='regularization', default=0.1, nargs='?',const=-1.0)
parser.add_argument('--rbf_width_factor',help='set RBF (radial basis function) width factor, default: 0.3; set this to -1 to crossvalidate when using --crossvalidate', type=float, dest='rbf_width_factor',default=0.3, nargs='?',const=0.3)
parser.add_argument('--svm_margin',help='set C parameter for SVC or SVR', const=-1.0,type=float, default=-1.0,nargs='?',dest='svm_margin')
parser.add_argument('--svm_epsilon',help='set svr epsilon parameter', const=-1.0,type=float, default=-1.0,nargs='?',dest='svm_epsilon')
parser.add_argument('--grid_size',help='grid size (if k: the map will be kxk, default k = sqrt(5*sqrt(Nfeatures))+2)', type=int, dest='grid_size',default=0)
parser.add_argument('--rbf_grid_size',help='RBF grid size (if m: the RBF grid will be mxm, default m = sqrt(grid_size))', type=int, dest='rbf_grid_size',default=0)
parser.add_argument('--n_neighbors',help='set number of neighbors for predictive modelling', const=1, type=int, default=1, nargs='?',dest='n_neighbors')
parser.add_argument('--random_state',help='change random state for map initialization (default is 5)', const=1234, type=int, default=1234, nargs='?',dest='random_state')
parser.add_argument('--representation',help='type of representation used for GTM: modes or means',dest='representation', const='modes', type=str, default='modes', nargs='?',choices=['means','modes'])
parser.add_argument('--kernel',help='type of kernel for Kernel GTM - default is cosine',dest='kernel', const='euclidean', type=str, default='euclidean', nargs='?',choices=['euclidean','laplacian','jaccard','cosine','linear'])
args = parser.parse_args()
print('')
print(args)
print('')


#################define html plotting functions
def plotHTML_any(label_numeric,label_names,coordinates,ids,plot_ids,modeltype="plot",useDiscrete=0):
	fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
	ax.grid(color='white', linestyle='solid')
	ax.set_title(modeltype,size=30)
	labels = ['point {0}'.format(i + 1) for i in range(label_names.shape[0])]
	scatter = ax.scatter(coordinates[:, 0],coordinates[:, 1], c=label_numeric, s=20,alpha=0.3,cmap=plt.cm.Spectral, edgecolor='black')
	if plot_ids:
		tooltip = mpld3.plugins.PointLabelTooltip(scatter,labels=["%s: label=%s" % t for t in zip(ids,label_names)])
	else:
		tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: label=%s" % t for t in zip(labels,label_names)])
	mpld3.plugins.connect(fig, tooltip)
	mpld3.save_html(fig,args.output+"_"+modeltype+".html")
	print("\nWrote html plot to disk: %s\n" % (args.output+"_"+modeltype+".html"))

def plotHTML_GTM(label_numeric,label_names,initialModel,optimizedModel,ids,plot_arrows=True,plot_ids=False,modeltype="GTM",useDiscrete=0):
	fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
	if useDiscrete:
		uGTM.plotClassMap(initialModel,optimizedModel,label_numeric)
	else:   
		uGTM.plotLandscape(initialModel,optimizedModel,label_numeric)
	means = optimizedModel.matMeans
	modes = optimizedModel.matModes
	ax.grid(color='white', linestyle='solid')
	ax.set_title(modeltype,size=30)
	labels = ['point {0}'.format(i + 1) for i in range(label_names.shape[0])]
	scatter = ax.scatter(means[:, 0],means[:, 1], c=label_numeric, s=20,alpha=0.3,cmap=plt.cm.Spectral, edgecolor='black')
	if plot_arrows:
		for i in range(label_names.shape[0]):
			plt.plot([means[i,0],modes[i,0]],[means[i,1],modes[i,1]],color='grey',linewidth=0.5)
	if plot_ids:
		tooltip = mpld3.plugins.PointLabelTooltip(scatter,labels=["%s: label=%s" % t for t in zip(ids,label_names)])
	else:
		tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: label=%s" % t for t in zip(labels,label_names)])
	mpld3.plugins.connect(fig, tooltip)
	mpld3.save_html(fig,args.output+"_"+modeltype+".html")
	print("\nWrote html plot to disk: %s\n" % (args.output+"_"+modeltype+".html"))

def plotHTML_GTM_withprojection(label_numeric,label_names,initialModel,optimizedModel,projections,ids,testids,plot_arrows=True,plot_ids=False,plot_test_ids=False,modeltype="GTM_projection",useDiscrete=0):
	fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
	if useDiscrete:
		uGTM.plotClassMap(initialModel,optimizedModel,label_numeric)
	else:
		uGTM.plotLandscape(initialModel,optimizedModel,label_numeric)
	means=projections.matMeans
	modes=projections.matModes
	ax.grid(color='white', linestyle='solid')
	ax.set_title(modeltype,size=30)
	labels = ['point {0}'.format(i + 1) for i in range(means.shape[0])]
	scatter = ax.scatter(means[:, 0],means[:, 1], c="black",s=20,alpha=1,edgecolor='black')
	if plot_arrows:
		for i in range(label_names.shape[0]):
			plt.plot([means[i,0],modes[i,0]],[means[i,1],modes[i,1]],color='grey',linewidth=0.5)
	if plot_ids:
		tooltip = mpld3.plugins.PointLabelTooltip(scatter,labels=list(testids))
	else:
		tooltip = mpld3.plugins.PointLabelTooltip(scatter,labels=labels)
	mpld3.plugins.connect(fig, tooltip)
	mpld3.save_html(fig,args.output+"_"+modeltype+".html")
	print("\nWrote html plot to disk: %s\n" % (args.output+"_"+modeltype+".html"))

#################process some of the arguments; make sure data is preprocessed if model is PCA
if args.model == 'PCA':
	args.pca = True
	args.n_components = 2
useDiscrete = 0
if args.labeltype == "discrete":
	useDiscrete = 1

if args.model and ((args.filenamedat and args.filenamelbls)):
	print("User provided model, data file and label names.")
	print("")
elif args.model and (args.usetest):
	print("User provided model and chose a default dataset.")
	print("")
else:
	print("Please provide model and data + labels or model + test data.")
	print("")
	exit

################load test examples if we choose to use default data from sklearn
if args.usetest == 's':
	matT,label = sklearn.datasets.samples_generator.make_s_curve(500, random_state=args.random_state)
elif args.usetest == 'swiss':
	matT,label = sklearn.datasets.make_swiss_roll(n_samples=2000,random_state=args.random_state)
elif args.usetest =='iris':
	iris = sklearn.datasets.load_iris()
	matT = iris.data
	label = iris.target_names[iris.target]
##############if it's not test data, then load provided data files
elif args.filenamedat and args.filenamelbls:
	matT = genfromtxt(args.filenamedat, delimiter=",", dtype=np.float64)
	if useDiscrete == 1:
		label = genfromtxt(args.filenamelbls, delimiter="\t", dtype=str)
	else:
		label = genfromtxt(args.filenamelbls, delimiter="\t", dtype=float)

#############save label name
savelabelname = np.copy(label)

############load ids for data points if there are provides
if args.filenameids is not None:
	ids = genfromtxt(args.filenameids, delimiter="\t", dtype=str)
else:
	ids = ""

###########define type of experiment
if (args.crossvalidate is True):
	type_of_experiment='crossvalidate'
elif (args.test is not None and useDiscrete==1 and args.model=='GTM'):
	type_of_experiment='traintest'
else:
	type_of_experiment='visualization'



#TYPE OF EXPERIMENT: 1: CROSSVALIDATION: CAN BE SVM, GTM, PCA
###################################################
###################################################
############# CROSSVALIDATION #####################
###################################################
###################################################

if (type_of_experiment =='crossvalidate'):
	uGTM.whichExperiment(matT,label,args,useDiscrete)
	exit


#TYPE OF EXPERIMENT: 2: TRAIN/TEST PREDICTION, CLASSIFICATION WITH GTM
###################################################
###################################################
################# TRAINTEST #######################
###################################################
###################################################


#########in case it's a train/test experiment for GTM classification
elif type_of_experiment == 'traintest':
	uniqClasses, labelnum = np.unique(label, return_inverse=True)
	test = genfromtxt(args.test, delimiter=",", dtype=np.float64)
	if args.filenametestids is not None:
		testids = genfromtxt(args.filenametestids, delimiter="\t", dtype=str)
	prediction=uGTM.advancedGTC(train=matT,labels=label,test=test,doPCA=args.pca,n_components=args.n_components,n_neighbors=args.n_neighbors,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size,predict_mode=args.predict_mode,prior=args.prior,regularization=args.regularization,rbf_width_factor=args.rbf_width_factor)
	plotHTML_GTM(label_numeric=labelnum,label_names=label,initialModel=prediction['initialModel'],optimizedModel=prediction['optimizedModel'],ids=ids,plot_arrows=True,plot_ids=args.filenameids,modeltype="GTM",useDiscrete=useDiscrete)
	uGTM.printClassPredictions(prediction,output=args.output)
	plotHTML_GTM_withprojection(label_numeric=labelnum,label_names=label,initialModel=prediction['initialModel'],optimizedModel=prediction['optimizedModel'],projections=prediction["indiv_projections"],ids=ids,testids=testids,plot_arrows=True,plot_ids=args.filenameids,plot_test_ids=args.filenametestids,modeltype="GTM_projection",useDiscrete=useDiscrete)
	exit



#TYPE OF EXPERIMENT: 3: VISUALIZATION: CAN BE t-SNE, GTM, PCA 
###################################################
###################################################
########### VISUALIZATION #########################
###################################################
###################################################


elif type_of_experiment == 'visualization':
	#convert labels to numeric
	if useDiscrete:
		uniqClasses, label = np.unique(label, return_inverse=True)

	#######preprocess 
	matT=uGTM.pcaPreprocess(matT=matT,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state)
	
	#set default parameters
	k=int(math.sqrt(5*math.sqrt(matT.shape[0])))+2
	m=int(math.sqrt(k))
	l=0.1
	s=0.3
	c=1000
	maxdim=100

	#set parameters if provided in options
	if args.regularization:
		l=args.regularization
	if args.rbf_width_factor:
		s=args.rbf_width_factor
	if args.grid_size:
		k=args.grid_size
	if args.rbf_grid_size:
		m=args.rbf_grid_size

	#########if it's for PCA visualization
	if args.model=='PCA':
		#make only html figure: change the code here if you want something else
		plotHTML_any(label_numeric=label,label_names=savelabelname,coordinates=matT,ids=ids,plot_ids=args.filenameids,modeltype="PCA")
		exit

	#########if it's for t-SNE visualization
	if args.model=='t-SNE':
		#perform tsne embedding
		tsne = manifold.TSNE(n_components=2, init='pca', random_state=args.random_state)
		matT_r = tsne.fit_transform(matT)
		plotHTML_any(label_numeric=label,label_names=savelabelname,coordinates=matT_r,ids=ids,plot_ids=args.filenameids,modeltype="t-SNE")
		exit

	#########if it's for GTM visualization
	if args.model=='GTM':
		#perform GTM embedding
		initialModel = uGTM.initialize(matT,k,m,s,random_state=args.random_state)
		start = time.time();
		optimizedModel = uGTM.optimize(matT,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);

		#make png with 1. GTM means viz 2. GTM modes viz 3. GTM landscape vis with mapping of points to nodes 4. GTM landscape viz
		uGTM.plotMultiPanelGTM(initialModel,optimizedModel,label,args.output,useDiscrete)	
		plotHTML_GTM(label_numeric=label,label_names=savelabelname,initialModel=initialModel,optimizedModel=optimizedModel,ids=ids,plot_arrows=True,plot_ids=args.filenameids,modeltype="GTM",useDiscrete=useDiscrete)
		exit

	###########if it's for kGTM visualization
	elif args.model=='kGTM':
		#kGTM embedding
		matK = uGTM.chooseKernel(matT,args.kernel)
		initialModel = uGTM.initializeKernel(matK,k,m,s,maxdim,random_state=args.random_state)
		start = time.time();
		optimizedModel = uGTM.optimizeKernel(matK,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
		#make png with 1. GTM means viz 2. GTM modes viz 3. GTM landscape vis with mapping of points to nodes 4. GTM landscape viz
		uGTM.plotMultiPanelGTM(initialModel,optimizedModel,label,args.output,useDiscrete)
		#interactive plot
		plotHTML_GTM(label_numeric=label,label_names=savelabelname,initialModel=initialModel,optimizedModel=optimizedModel,ids=ids,plot_arrows=True,plot_ids=args.filenameids,modeltype="kGTM",useDiscrete=useDiscrete)
		exit

	##########if it's to compare GTM, PCA, LLE and t_SNE visualizations
	elif args.model=='compare': 
		print("Computing GTM embedding")
		initialModel = uGTM.initialize(matT,k,m,s,random_state=args.random_state)
		start = time.time();
		optimizedModel = uGTM.optimize(matT,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
		fig = plt.figure(figsize=(12,10))
		ax = fig.add_subplot(331)
		ax.scatter(matT[:,0], matT[:,1], c=label, cmap=plt.cm.Spectral)
		plt.axis('tight')
		plt.xticks([]), plt.yticks([])
		if args.pca:
			plt.title('PCA')
		else:
			plt.title('Original data')
		ax = fig.add_subplot(334)
		ax.scatter(optimizedModel.matMeans[:, 0], optimizedModel.matMeans[:, 1], c=label, cmap=plt.cm.Spectral)
		plt.axis('tight')
		plt.xticks([]), plt.yticks([])
		plt.title('GTM')
		ax = fig.add_subplot(337)
		if useDiscrete:
			uGTM.plotClassMap(initialModel,optimizedModel,label)
		else:
			uGTM.plotLandscape(initialModel,optimizedModel,label)
		matK = uGTM.chooseKernel(matT,'laplacian') 
		print("Computing kGTM embedding (laplacian)")
		initialModel = uGTM.initializeKernel(matK,k,m,s,maxdim,random_state=args.random_state)
		print("The estimated feature space dimension is: ", initialModel.nDimensions)
		start = time.time();
		optimizedModel = uGTM.optimizeKernel(matK,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for kGTM: ",elapsed);
		ax = fig.add_subplot(335)
		ax.scatter(optimizedModel.matMeans[:, 0], optimizedModel.matMeans[:, 1], c=label, cmap=plt.cm.Spectral)
		plt.axis('tight')
		plt.xticks([]), plt.yticks([])
		plt.title('kGTM (laplacian)')
		ax = fig.add_subplot(338)
		if useDiscrete:
			uGTM.plotClassMap(initialModel,optimizedModel,label)
		else:
			uGTM.plotLandscape(initialModel,optimizedModel,label)
		matK = uGTM.chooseKernel(matT,'euclidean')
		print("Computing kGTM embedding (euclidean)")
		initialModel = uGTM.initializeKernel(matK,k,m,s,maxdim,random_state=args.random_state)
		print("The estimated feature space dimension is: ", initialModel.nDimensions)
		start = time.time();
		optimizedModel = uGTM.optimizeKernel(matK,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for kGTM: ",elapsed);
		ax = fig.add_subplot(336)
		ax.scatter(optimizedModel.matMeans[:, 0], optimizedModel.matMeans[:, 1], c=label, cmap=plt.cm.Spectral)
		plt.axis('tight')
		plt.xticks([]), plt.yticks([])
		plt.title('kGTM (euclidean)')
		ax = fig.add_subplot(339)
		if useDiscrete:
			uGTM.plotClassMap(initialModel,optimizedModel,label)
		else:
			uGTM.plotLandscape(initialModel,optimizedModel,label)
		print("Computing LLE embedding")
		start = time.time(); 
		matT_r, err = manifold.locally_linear_embedding(matT, n_neighbors=12,n_components=2)
		end = time.time(); elapsed = end - start; print("time taken for LLE: ",elapsed);
		print("Done. Reconstruction error: %g" % err)
		ax = fig.add_subplot(332)
		ax.scatter(matT_r[:, 0], matT_r[:, 1], c=label, cmap=plt.cm.Spectral)
		plt.axis('tight')
		plt.xticks([]), plt.yticks([])
		plt.title('LLE')
		print("Computing t-SNE: embedding")
		start = time.time();
		tsne = manifold.TSNE(n_components=2, init='pca', random_state=args.random_state)
		matT_r = tsne.fit_transform(matT)
		end = time.time(); elapsed = end - start; print("time taken for TSNE: ",elapsed);
		print("Done. Reconstruction error: %g" % err)
		ax = fig.add_subplot(333)
		ax.scatter(matT_r[:, 0], matT_r[:, 1], c=label, cmap=plt.cm.Spectral)
		plt.axis('tight')
		plt.xticks([]), plt.yticks([])
		plt.title('t-SNE')
		fig.savefig(args.output) 
		plt.close(fig)
		exit
	else:
		print("Sorry. Model not recognized")
		exit
else:
	print("Sorry. Could not guess what you wanted. Remember to define --model and (--data and --labels) or --model and --usetest.")
	exit
