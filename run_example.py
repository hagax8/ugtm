import uGTM;
import numpy as np;
import time;
import matplotlib.pyplot as plt
import sklearn.datasets;
from sklearn.cluster import AgglomerativeClustering;
from sklearn.decomposition import PCA;
import mpl_toolkits.mplot3d.axes3d as p3;
import argparse
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='data file, csv format', dest='filenamedat')
parser.add_argument('--labels', help='label file, csv format', dest='filenamelbls')
args = parser.parse_args()

#parameters;
k=10;
m=4;
l=0.1;
s=1;
c=100;

#process data
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
label = data - 1
matT = sklearn.preprocessing.scale(matT,axis=0, with_mean=True, with_std=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(matT)
matT = scaler.transform(matT)


#run model
print("Computing GTM embedding")
#initialModel: gaussian mixture and swiss roll
initialModel = uGTM.initiliaze(matT,k,m,s)
#optimizedModel: gaussian mixture and swiss roll
start = time.time();
optimizedModel = uGTM.optimize(matT,initialModel,l,c)
end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);
#plot and compare with LLE
mm = optimizedModel.matMeans
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.scatter(matT[:, 0],matT[:, 1], matT[:, 2], c=label, cmap=plt.cm.Spectral)
ax.set_title("Original data")
ax = fig.add_subplot(222)
ax.scatter(mm[:, 0], mm[:, 1], c=label, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data: GTM')
#plt.show()

#compare with LLE

from sklearn import manifold

print("Computing LLE embedding")
start = time.time(); 
matT_r, err = manifold.locally_linear_embedding(matT, n_neighbors=12,n_components=2)
end = time.time(); elapsed = end - start; print("time taken for LLE: ",elapsed);
print("Done. Reconstruction error: %g" % err)
ax = fig.add_subplot(223)
ax.scatter(matT_r[:, 0], matT_r[:, 1], c=label, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data: LLE')
#plt.show()
print("Computing t-SNE: embedding")
start = time.time();
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
matT_r = tsne.fit_transform(matT)
end = time.time(); elapsed = end - start; print("time taken for TSNE: ",elapsed);
print("Done. Reconstruction error: %g" % err)
ax = fig.add_subplot(224)
ax.scatter(matT_r[:, 0], matT_r[:, 1], c=label, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data: t-SNE')
#plt.show()


fig.savefig('./fig.png')   # save the figure to file
plt.close(fig)
