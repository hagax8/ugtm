import uGTM;
import numpy as np;
import time;
import matplotlib.pyplot as plt
import sklearn.datasets;
from sklearn.cluster import AgglomerativeClustering;
from sklearn.decomposition import PCA;
import mpl_toolkits.mplot3d.axes3d as p3;
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='data file, csv format', dest='filenamedat')
parser.add_argument('--labels', help='label file, csv format', dest='filenamelbls')
args = parser.parse_args()

#parameters;
k=15;
m=4;
l=0.1;
s=1;

#process data
import csv
raw_data = open(args.filenamedat, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
matT = np.array(x).astype('float')
raw_labels = open(args.filenamelbls, 'rt')
reader = csv.reader(raw_labels, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
label = np.array(x).astype('int') - 1

#minmax scale
matT = sklearn.preprocessing.scale(matT,axis=0, with_mean=True, with_std=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(matT)
matT = scaler.transform(matT)


#run model
print("Computing GTM embedding")
initialModel = uGTM.initiliaze(matT,k,m,s,l)
start = time.time();
optimizedModel = uGTM.optimize(matT,initialModel,l,100)
end = time.time(); elapsed = end - start; print("time taken for GTM: ",elapsed);


#plot
mm = optimizedModel.matMeans
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(matT[:, 0],matT[:, 1], matT[:, 2], c=label, cmap=plt.cm.Spectral)
ax.set_title("Original data")
ax = fig.add_subplot(122)
ax.scatter(mm[:, 0], mm[:, 1], c=label, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data: GTM')

fig.savefig('./fig.png')   # save the figure to file
plt.close(fig)
