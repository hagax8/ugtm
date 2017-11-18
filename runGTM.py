import uGTM;
import numpy as np;
import time;
import matplotlib
import matplotlib.pyplot as plt, mpld3
import sklearn.datasets;
from sklearn.cluster import AgglomerativeClustering;
from sklearn.decomposition import PCA;
from sklearn.metrics import pairwise_distances
import mpl_toolkits.mplot3d.axes3d as p3;
import argparse
import scipy
import math

np.random.rand(4)

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='data file in csv format without header (must be a similarity matrix if --model kGTM is set, otherwise not)', dest='filenamedat')
parser.add_argument('--labels', help='label file in csv format without header', dest='filenamelbls')
parser.add_argument('--labeltype', help='{continuous, discrete}: labels are continuous or discrete', dest='labeltype')
parser.add_argument('--usetest', help='{s, swiss, iris): use S or swiss roll or iris test data', dest='usetest')
parser.add_argument('--model',help='{GTM, kGTM, compare}: GTM model, kernel GTM model, or comparison between GTM, kGTM, LLE and tSNE', dest='model') 
parser.add_argument('--output',help='output name', dest='output')
parser.add_argument('--optimize',help='show best l (regularization coefficient) and w combination (RBF width factor) for classification or regression',dest='optimize', action='store_true')
args = parser.parse_args()



useDiscrete = 0
if args.labeltype == "discrete":
	useDiscrete = 1
if args.usetest == 's':
	matT,label = sklearn.datasets.samples_generator.make_s_curve(500, random_state=0)
elif args.usetest == 'swiss':
	matT,label = sklearn.datasets.make_swiss_roll(n_samples=1000,random_state=0)
elif args.usetest =='iris':
	iris = sklearn.datasets.load_iris()
	matT = iris.data
	label = iris.target
else:
	import csv
	raw_data = open(args.filenamedat, 'rt')
	reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
	x = list(reader)
	data = np.array(x).astype('float')
	matT = data
	raw_labels = open(args.filenamelbls, 'rt')
	reader = csv.reader(raw_labels, delimiter=',', quoting=csv.QUOTE_NONE)
	x = list(reader)
	data = np.array(x).astype('int')
	label = data[:,0]


k=int(math.sqrt(5*math.sqrt(matT.shape[0])))+2
m=int(math.sqrt(k))
l=0.1
s=1
c=100
maxdim=200


#minmax scale
if (args.optimize is not None) and useDiscrete==1 and args.model=='GTM':
	uGTM.optimizeGTC(matT,label)
#	uGTM.optimizeSVCrbf(matT,label)
#	uGTM.optimizeSVC(matT,label)
#matT = sklearn.preprocessing.scale(matT,axis=0, with_mean=True, with_std=True)

elif(args.optimize is not None) and useDiscrete==0 and args.model=='GTM':
	uGTM.optimizeGTR(matT,label)
elif(args.optimize is not None) and useDiscrete==1 and args.model=='compare':
	uGTM.optimizeSVCrbf(matT,label)
	uGTM.optimizeSVC(matT,label)
	uGTM.optimizeGTC(matT,label)
elif(args.optimize is not None) and useDiscrete==0 and args.model=='compare':
	uGTM.optimizeSVR(matT,label)
	uGTM.optimizeGTR(matT,label)
else:
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler.fit(matT)
	matT = scaler.transform(matT)

	if args.model=='GTM':
		initialModel = uGTM.initialize(matT,k,m,s)
		start = time.time();
		optimizedModel = uGTM.optimize(matT,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
		fig = plt.figure(figsize=(10,10))
		means=optimizedModel.matMeans
		modes=optimizedModel.matModes
		#prediction=uGTM.predictNN(initialModel,optimizedModel,label,matT,'classification',1,'modes')
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
		ax.set_title("Map",size=30)
		labels = ['point {0}'.format(i + 1) for i in range(label.shape[0])]
		scatter = ax.scatter(means[:, 0],means[:, 1], c=label, s=20,alpha=0.3,cmap=plt.cm.Spectral, edgecolor='black')
		for i in range(label.shape[0]):
			plt.plot([means[i,0],modes[i,0]],[means[i,1],modes[i,1]],color='grey',linewidth=0.5)
		#tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: -logP=%f" % t for t in zip(labels, list(-np.log10(z)))])
		tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=["%s: label=%s" % t for t in zip(labels, label)])
		mpld3.plugins.connect(fig, tooltip)
		mpld3.show()
	elif args.model=='kGTM':
		initialModel = uGTM.initializeKernel(matT,k,m,s,maxdim)
		start = time.time();
		optimizedModel = uGTM.optimizeKernel(matT,initialModel,l,c)
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
		initialModel = uGTM.initialize(matT,k,m,s)
		start = time.time();
		optimizedModel = uGTM.optimize(matT,initialModel,l,c)
		end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
		fig = plt.figure(figsize=(12,10))
		ax = fig.add_subplot(331, projection='3d')
		ax.scatter(matT[:, 0],matT[:, 1], matT[:, 2], c=label, cmap=plt.cm.Spectral)
		ax.set_title("Original data")
		plt.axis('tight')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
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
		matK = sklearn.metrics.pairwise.laplacian_kernel(matT)
		matK
		print("Computing kGTM embedding")
		initialModel = uGTM.initializeKernel(matK,k,m,s,maxdim)
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
		matK = np.divide(1,(1+pairwise_distances(matT, metric="euclidean")))
		print("Computing kGTM embedding")
		initialModel = uGTM.initializeKernel(matK,k,m,s,maxdim)
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
		from sklearn import manifold
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
		tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
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

