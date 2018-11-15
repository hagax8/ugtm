"""Preprocessing operations (mostly using scikit-learn functions).
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KernelCenterer
from sklearn.decomposition import PCA


class ProcessedTrainTest(object):
    r"""Class for processed train and test set.

    Arguments
    =========
    train : array of shape (n_train, n_dimensions)
        Train data matrix.
    test : array of shape (n_test, ndimensions)
        Test data matrix.
    """

    def __init__(self, train, test):
        r""" Constructor for :class:`~ugtm.ugtm_preprocess.ProcessedTrainTest`.

        Parameters
        =========
        train : array of shape (n_train, n_dimensions)
            Train data matrix.
        test : array of shape (n_test, ndimensions)
            Test data matrix.
        """
        self.train = train
        self.test = test


def pcaPreprocess(data, doPCA=False, n_components=-1, missing=False,
                  missing_strategy='median', random_state=1234):
    r"""Preprocess data using PCA.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    n_components : int, optional (default = -1)
        Number of components for PCA pre-processing.
        If set to -1, keep principal components
        accounting for 80% of data variance.
    missing : bool, optional (default = True)
        Replace missing values (calls scikit-learn functions).
    missing_strategy : str (default = 'median')
        Scikit-learn missing data strategy.
    random_state : int (default = 1234)
        Random state.

    Returns
    =======
    array of shape (n_individuals, n_components)
        Data projected onto principal axes.
    """
    if missing:
        imp = SimpleImputer(strategy=missing_strategy)
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
                     missing_strategy='median', random_state=1234):
    r"""Preprocess train and test data using PCA.

    Parameters
    ==========
    train : array of shape (n_individuals, n_train)
        Train data matrix.
    test : array of shape (n_individuals, n_test)
        Test data matrix.
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    n_components : int, optional (default = -1)
        Number of components for PCA pre-processing.
        If set to -1, keep principal components
        accounting for 80% of data variance.
    missing : bool, optional (default = True)
        Replace missing values (calls scikit-learn functions).
    missing_strategy : str (default = 'median')
        Scikit-learn missing data strategy.
    random_state : int (default = 1234)
        Random state.

    Returns
    =======
    instance of :class:`~ugtm.ugtm_preprocess.ProcessedTrainTest`
    """
    if missing:
        imp = SimpleImputer(strategy=missing_strategy)
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
    r"""Kernalize data (uses sklearn)

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Data matrix.
    kerneltype : {'euclidean', 'cosine', 'laplacian', 'polynomial_kernel', 'jaccard'}, optional
        Kernel type.

    Returns
    =======
    array of shape (n_individuals, n_individuals)
    """
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
