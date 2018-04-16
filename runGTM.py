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


parser = argparse.ArgumentParser(description='Generate and assess GTM maps for classification or regression.')
parser.add_argument('--data', help='data file in csv format without header (must be a similarity matrix if --model kGTM is set, otherwise not)', dest='filenamedat')
parser.add_argument('--labels', help='label file in csv format without header', dest='filenamelbls')
parser.add_argument('--ids', help='label file in csv format without header', dest='filenameids')
parser.add_argument('--labeltype', help='labels are continuous or discrete', dest='labeltype', choices=['continuous','discrete'])
parser.add_argument('--usetest', help='use S or swiss roll or iris test data', dest='usetest',choices=['s','swiss','iris'])
parser.add_argument('--model',help='GTM model, kernel GTM model, SVM, PCA or comparison between: GTM, kGTM, LLE and tSNE for simple visualization, GTM and SVM for regression or classification (when --optimize is set); the optimized parameters for GTM are regularization and rbf_width_factor for given grid_size and rbf_grid_size', dest='model',choices=['GTM','kGTM','SVM','PCA','t-SNE','compare']) 
parser.add_argument('--output',help='output name', dest='output')
parser.add_argument('--optimize',help='show best l (regularization coefficient) and w combination (RBF width factor) for classification or regression, with default grid size parameter k = sqrt(5*sqrt(Nfeatures))+2) and RBF grid size parameter m = sqrt(k)', action='store_true' )
parser.add_argument('--pca', help='do PCA preprocessing; if --n_components is not set, will use number of PCs explaining 80%% of variance', action='store_true')
parser.add_argument('--missing', help='there is missing data (encoded by NA)', action='store_true')
parser.add_argument('--missing_strategy',help='missing data strategy, missing values encoded by NA; default is median', const='median', type=str, default='median', nargs='?',dest='missing_strategy',choices=['mean','median','most_frequent'])
parser.add_argument('--n_components',help='set number of components for PCA pre-processing, if --pca flag is present', const=-1, type=int, default=-1, nargs='?',dest='n_components')

parser.add_argument('--regularization',help='set regularization factor, default: 0.1', type=float, dest='regularization')
parser.add_argument('--rbf_width_factor',help='set RBF (radial basis function) width factor, default: 1.0', type=float, dest='rbf_width_factor')
parser.add_argument('--grid_size',help='grid size (if k: the map will be kxk, default k = sqrt(5*sqrt(Nfeatures))+2)', type=int, dest='grid_size',default=0)
parser.add_argument('--rbf_grid_size',help='RBF grid size (if m: the RBF grid will be mxm, default m = sqrt(grid_size))', type=int, dest='rbf_grid_size',default=0)

parser.add_argument('--n_neighbors',help='set number of neighbors for predictive modelling', const=1, type=int, default=1, nargs='?',dest='n_neighbors')
parser.add_argument('--random_state',help='change random state for map initialization (default is 5)', const=1234, type=int, default=1234, nargs='?',dest='random_state')
parser.add_argument('--representation',help='type of representation used for GTM: modes or means',dest='representation', const='modes', type=str, default='modes', nargs='?',choices=['means','modes'])
parser.add_argument('--kernel',help='type of kernel for Kernel GTM - default is cosine',dest='kernel', const='euclidean', type=str, default='euclidean', nargs='?',choices=['euclidean','laplacian','jaccard','cosine','linear'])
#parser.add_argument('--representation',help='{modes, means} Type of representation used for GTM: modes or means',dest='representation')
args = parser.parse_args()
print(args)


if args.model == 'PCA':
	args.pca = True
	args.n_components = 2

useDiscrete = 0
doOptim = False
if args.labeltype == "discrete":
	useDiscrete = 1
if args.usetest == 's':
	matT,label = sklearn.datasets.samples_generator.make_s_curve(500, random_state=args.random_state)
elif args.usetest == 'swiss':
	matT,label = sklearn.datasets.make_swiss_roll(n_samples=1500,random_state=args.random_state)
elif args.usetest =='iris':
	iris = sklearn.datasets.load_iris()
	matT = iris.data
	label = iris.target
elif args.filenamedat and args.filenamelbls:
	import csv
	#raw_data = open(args.filenamedat, 'rt')
	#reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
	#x = list(reader)
	#data = np.array(x).astype('float')
	#matT = data
	matT = genfromtxt(args.filenamedat, delimiter=",", dtype=np.float64)
	#raw_labels = open(args.filenamelbls, 'rt')
	#reader = csv.reader(raw_labels, delimiter=',', quoting=csv.QUOTE_NONE)
	#x = list(reader)
	#data = np.array(x).astype('str')
	if useDiscrete == 1:
		label = genfromtxt(args.filenamelbls, delimiter="\t", dtype=str)
	else:
		label = genfromtxt(args.filenamelbls, delimiter="\t", dtype=float)

savelabelname = np.copy(label)
if useDiscrete:
	uniqClasses, label = np.unique(label, return_inverse=True)

if args.filenameids is not None:
	ids = genfromtxt(args.filenameids, delimiter="\t", dtype=str)

if ((args.filenamedat and args.filenamelbls) or args.usetest) and len(matT[0,:]) > 100:
	args.pca = True

if (args.optimize is True) and useDiscrete==1 and args.model=='GTM':
        uGTM.optimizeGTC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,n_neighbors=args.n_neighbors,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size)
    #	uGTM.optimizeGTC(matT,label,args.pca,args.n_components)
#	uGTM.optimizeSVCrbf(matT,label)
#	uGTM.optimizeSVC(matT,label)
#matT = sklearn.preprocessing.scale(matT,axis=0, with_mean=True, with_std=True)
elif(args.optimize is True) and useDiscrete==0 and args.model=='GTM':
	uGTM.optimizeGTR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,n_neighbors=args.n_neighbors,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size)
#	uGTM.optimizeGTR(matT=matT,labels=label,doPCA=args.pca)
elif(args.optimize is True) and useDiscrete==1 and args.model=='compare':
	uGTM.optimizeSVCrbf(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state)
	uGTM.optimizeSVC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state)
	uGTM.optimizeGTC(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size)
elif(args.optimize is True) and useDiscrete==0 and args.model=='compare':
	uGTM.optimizeSVR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state)
	uGTM.optimizeGTR(matT=matT,labels=label,doPCA=args.pca,n_components=args.n_components,n_neighbors=args.n_neighbors,representation=args.representation,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state,k=args.grid_size,m=args.rbf_grid_size)
elif args.model and ((args.filenamedat and args.filenamelbls) or args.usetest):
	matT=uGTM.pcaPreprocess(matT=matT,doPCA=args.pca,n_components=args.n_components,missing=args.missing,missing_strategy=args.missing_strategy,random_state=args.random_state)
	k=int(math.sqrt(5*math.sqrt(matT.shape[0])))+2
	m=int(math.sqrt(k))
	l=0.01
	s=1
	c=1000
	maxdim=100

	if args.regularization:
		l=args.regularization

	if args.rbf_width_factor:
		s=args.rbf_width_factor

	if args.grid_size:
		k=args.grid_size

	if args.rbf_grid_size:
		m=args.rbf_grid_size

	if args.model=='PCA':
		fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
		ax.grid(color='white', linestyle='solid')
		ax.set_title("PCA",size=30)
		labels = ['point {0}'.format(i + 1) for i in range(label.shape[0])]
		scatter = ax.scatter(matT[:, 0],matT[:, 1], c=label, s=20,alpha=0.3,cmap=plt.cm.Spectral, edgecolor='black')
		if args.filenameids is not None:
			tooltip = mpld3.plugins.PointLabelTooltip(scatter,labels=["%s: label=%s" % t for t in zip(ids,savelabelname)])
		else:
			tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: label=%s" % t for t in zip(labels, savelabelname)])
		mpld3.plugins.connect(fig, tooltip)
		mpld3.save_html(fig,args.output+"_pca.html")
	if args.model=='t-SNE':
		fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
		ax.grid(color='white', linestyle='solid')
		ax.set_title("t-SNE",size=30)
		labels = ['point {0}'.format(i + 1) for i in range(label.shape[0])]
		tsne = manifold.TSNE(n_components=2, init='pca', random_state=args.random_state)
		matT_r = tsne.fit_transform(matT)
		scatter = ax.scatter(matT_r[:, 0],matT_r[:, 1], c=label, s=20,alpha=0.3,cmap=plt.cm.Spectral, edgecolor='black')
		if args.filenameids is not None:
			tooltip = mpld3.plugins.PointLabelTooltip(scatter,labels=["%s: label=%s" % t for t in zip(ids,savelabelname)])
		else:
			tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: label=%s" % t for t in zip(labels, savelabelname)])
		mpld3.plugins.connect(fig, tooltip)
		mpld3.save_html(fig,args.output+"_t-sne.html")
	if args.model=='GTM':
		initialModel = uGTM.initialize(matT,k,m,s,random_state=args.random_state)
		start = time.time();
		optimizedModel = uGTM.optimize(matT,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
		fig = plt.figure(figsize=(10,10))
		means=optimizedModel.matMeans
		modes=optimizedModel.matModes
		prediction=uGTM.predictNN(initialModel,optimizedModel,label,matT,'classification',1,'means')
		ax = fig.add_subplot(221); ax.scatter(means[:, 0], means[:, 1], c=label, cmap=plt.cm.Spectral); plt.axis('tight'); plt.xticks([]), plt.yticks([]); plt.title('Means'); 
		ax = fig.add_subplot(222); ax.scatter(modes[:, 0], modes[:, 1], c=label, cmap=plt.cm.Spectral); plt.axis('tight'); plt.xticks([]), plt.yticks([]); plt.title('Modes');
		ax = fig.add_subplot(223);
		if useDiscrete:
			uGTM.plotClassMap(initialModel,optimizedModel,label)
		else:
			uGTM.plotLandscape(initialModel,optimizedModel,label)
		for i in range(label.shape[0]):
			plt.plot([means[i,0],modes[i,0]],[means[i,1],modes[i,1]],color='grey',linewidth=0.5) 
		ax = fig.add_subplot(224);
		if useDiscrete:
			uGTM.plotClassMap(initialModel,optimizedModel,label)
		else:
			uGTM.plotLandscape(initialModel,optimizedModel,label)
		fig.savefig(args.output) 
		plt.close(fig)
		fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
		#scatter = ax.scatter(optimizedModel.matMeans[:, 0],optimizedModel.matMeans[:, 1], c=-np.log10(z), s= 100 * -np.log10(z),alpha=0.3,cmap='hot')
		#scatter = ax.scatter(means[:, 0],means[:, 1], c=label, s=20,alpha=0.3,cmap=plt.cm.Spectral)
		if useDiscrete:
			uGTM.plotClassMap(initialModel,optimizedModel,label)
		else:
			uGTM.plotLandscape(initialModel,optimizedModel,label)	
		ax.grid(color='white', linestyle='solid')
		ax.set_title("GTM",size=30)
		labels = ['point {0}'.format(i + 1) for i in range(label.shape[0])]
		scatter = ax.scatter(means[:, 0],means[:, 1], c=label, s=20,alpha=0.3,cmap=plt.cm.Spectral, edgecolor='black')
		for i in range(label.shape[0]):
			plt.plot([means[i,0],modes[i,0]],[means[i,1],modes[i,1]],color='grey',linewidth=0.5)
		#tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: -logP=%f" % t for t in zip(labels, list(-np.log10(z)))])
		#tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: label=%s" % t for t in zip(labels, savelabelname)])
		if args.filenameids is not None:
			tooltip = mpld3.plugins.PointLabelTooltip(scatter,labels=["%s: label=%s" % t for t in zip(ids,savelabelname)]) 
		else:
			tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: label=%s" % t for t in zip(labels, savelabelname)])
		mpld3.plugins.connect(fig, tooltip)
		mpld3.save_html(fig,args.output+"_GTM.html")
		#mpld3.show()
	elif args.model=='kGTM':
		matK = uGTM.chooseKernel(matT,args.kernel)
		initialModel = uGTM.initializeKernel(matK,k,m,s,maxdim,random_state=args.random_state)
		start = time.time();
		optimizedModel = uGTM.optimizeKernel(matK,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
		fig = plt.figure(figsize=(10,10))
		means=optimizedModel.matMeans
		modes=optimizedModel.matModes
		ax = fig.add_subplot(221); ax.scatter(means[:, 0], means[:, 1], c=label, cmap=plt.cm.Spectral);
		ax = fig.add_subplot(222); ax.scatter(modes[:, 0], modes[:, 1], c=label, cmap=plt.cm.Spectral);
		ax = fig.add_subplot(223); uGTM.plotClassMap(initialModel,optimizedModel,label)
		fig.savefig(args.output)
		plt.close(fig)
	elif args.model=='compare': 
		print("Computing GTM embedding")
		initialModel = uGTM.initialize(matT,k,m,s,random_state=args.random_state)
		start = time.time();
		optimizedModel = uGTM.optimize(matT,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
		fig = plt.figure(figsize=(12,10))
		#ax = fig.add_subplot(331, projection='3d')
		#ax.scatter(matT[:, 0],matT[:, 1], matT[:, 2], c=label, cmap=plt.cm.Spectral)
		#ax.set_title("Original data")
		#plt.axis('tight')
		#plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
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
		#plot and compare with LLE
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
		#matK = np.divide(1,(1+pairwise_distances(matT, metric="euclidean")))
		matK = uGTM.chooseKernel(matT,'euclidean')
		print("Computing kGTM embedding (euclidean)")
		initialModel = uGTM.initializeKernel(matK,k,m,s,maxdim,random_state=args.random_state)
		print("The estimated feature space dimension is: ", initialModel.nDimensions)
		start = time.time();
		optimizedModel = uGTM.optimizeKernel(matK,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for kGTM: ",elapsed);
		#plot and compare with LLE
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

