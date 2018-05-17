from __future__ import print_function
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KernelCenterer
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer


class ProcessedTrainTest(object):
    def __init__(self, train, test):
        self.train = train
        self.test = test


def pcaPreprocess(data, doPCA=False, n_components=-1, missing=False,
                  missing_strategy='most_frequent', random_state=1234):
    if missing:
        imp = Imputer(strategy=missing_strategy, axis=0)
        data = imp.fit_transform(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if n_components == -1:
        n_components = 0.80
    if doPCA:
        pca = PCA(random_state=random_state, n_components=n_components)
        data = pca.fit_transform(data)
        n_components = pca.n_components_
        print("Used %s components explaining %s%% of the variance\n" %
              (n_components,
               pca.explained_variance_ratio_.cumsum()[n_components-1]*100))
    return(data)


def processTrainTest(train, test, doPCA, n_components, missing=False,
                     missing_strategy='most_frequent', random_state=1234):
    if missing:
        imp = Imputer(strategy=missing_strategy, axis=0)
        train = imp.fit_transform(train)
        test = imp.transform(test)
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    if(n_components == -1):
        n_components = 0.80
    if doPCA:
        pca = PCA(random_state=random_state, n_components=n_components)
        train = pca.fit_transform(train)
        test = pca.transform(test)
    return(ProcessedTrainTest(train, test))


def chooseKernel(data, kerneltype='euclidean'):
    if kerneltype == 'euclidean':
        K = np.divide(1, (1+pairwise_distances(data, metric="euclidean")))
    elif kerneltype == 'cosine':
        K = (pairwise.cosine_kernel(data))
    elif kerneltype == 'laplacian':
        K = (pairwise.laplacian_kernel(data))
    elif kerneltype == 'linear':
        K = (pairwise.linear_kernel(data))
    elif kerneltype == 'polynomial_kernel':
        K = (pairwise.polynomial_kernel(data))
    elif kerneltype == 'jaccard':
        K = 1-distance.cdist(data, data, metric='jaccard')
    scaler = KernelCenterer().fit(K)
    return(scaler.transform(K))
