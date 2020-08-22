"""GTM transformer, classifier and regressor compatible with sklearn
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import TransformerMixin
from . import ugtm_gtm
from . import ugtm_landscape
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import NearestNeighbors
import numpy as np


class eGTM(BaseEstimator, TransformerMixin):
    """eGTM: GTM Transformer for sklearn pipeline.

    Arguments
    =========
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        m is the qrt of the number of RBF centers.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
    s : float, optional (default = 0.3)
        RBF width factor.
        One of four GTM hyperparameters (k, m, s, regul).
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = 0.1)
        One of four GTM hyperparameters (k, m, s, regul).
        Regularization coefficient.
    random_state : int (default = 1234)
        Random state.
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    verbose : bool, optional (default = False)
        Verbose mode (outputs loglikelihood values during EM algorithm).
    model : {'means', 'modes', 'responsibilities','complete'}, optional
        GTM data representations:
        'means' for mean data positions,
        'modes' for positions with  max. responsibilities,
        'responsibilities' for probability distribution on the map,
        'complete' for a complete instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`

    """

    def __init__(self, k=16, m=4, s=0.3, regul=0.1,
                 random_state=1234,
                 niter=200, verbose=False, model="means"):
        """Constructor for eGTM class.

        Parameters
        ==========
        k : int, optional (default = 16)
            If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
            k is the sqrt of the number of GTM nodes.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
        m : int, optional (default = 4)
            If m is set to 0, m is computed as sqrt(k).
            m is the qrt of the number of RBF centers.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
        s : float, optional (default = 0.3)
            RBF width factor.
            One of four GTM hyperparameters (k, m, s, regul).
            Parameter to tune width of RBF functions.
            Impacts manifold flexibility.
        regul : float, optional (default = 0.1)
            One of four GTM hyperparameters (k, m, s, regul).
            Regularization coefficient.
        random_state : int (default = 1234)
            Random state.
        niter : int, optional (default = 200)
            Number of iterations for EM algorithm.
        verbose : bool, optional (default = False)
            Verbose mode (outputs loglikelihood values during EM algorithm).
        model : {'means', 'modes', 'responsibilities','complete'}, optional
            GTM data representations:
            'means' for mean data positions,
            'modes' for positions with  max. responsibilities,
            'responsibilities' for probability distribution on the map,
            'complete' for a complete instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`

        """
        assert model in ('means', 'modes', 'responsibilities', 'complete'),\
            "model must be either of 'means', 'modes', 'responsibilities', or 'complete'"
        self.k = k
        self.m = m
        self.s = s
        self.regul = regul
        self.random_state = random_state
        self.niter = niter
        self.verbose = verbose
        self.model = model

    def fit(self, X, y=None):
        """Fits GTM to X using :class:`~ugtm.ugtm_classes.OptimizedGTM`.

        Parameters
        ==========

        X : 2D array
            Data matrix.

        """
        X = check_array(X)

        self.initialModel = ugtm_gtm.initialize(X, self.k,
                                                self.m, self.s,
                                                self.random_state)
        self.optimizedModel = ugtm_gtm.optimize(X, self.initialModel,
                                                self.regul,
                                                self.niter,
                                                verbose=self.verbose)

        return self

    def transform(self, X):
        """Projects new data X onto GTM using :func:`~ugtm.ugtm_gtm.projection`.

        Parameters
        ==========

        X : 2D array
            Data matrix.

        Returns
        =======
        if self.model="means", array of shape (n_instances, 2),
        if self.model="modes", array of shape (n_instances, 2),
        if self.model="responsibilities", array of shape (n_instances, n_nodes),
        if self.model="complete", instance of class :class:`~ugtm.ugtm_classes.OptimizedGTM`
        """

        # Check fit
        check_is_fitted(self, ['optimizedModel'])

        # Input validation
        X = check_array(X)

        # Project new data onto fitted GTM
        self.projected = ugtm_gtm.projection(self.optimizedModel, X)

        # Output
        dic = {}
        dic["complete"] = self.projected
        dic["means"] = self.projected.matMeans
        dic["modes"] = self.projected.matModes
        dic["responsibilities"] = self.projected.matR

        return dic[self.model]

    def fit_transform(self, X, y=None):
        """Fits and transforms X using GTM.

        Parameters
        ==========

        X : 2D array
            Data matrix.

        Returns
        =======
        if self.model="means", array of shape (n_instances, 2),
        if self.model="modes", array of shape (n_instances, 2),
        if self.model="responsibilities", array of shape (n_instances, n_nodes),
        if self.model="complete", instance of class :class:`~ugtm.ugtm_classes.OptimizedGTM`
        """

        X = check_array(X)

        self.initialModel = ugtm_gtm.initialize(X, self.k,
                                                self.m, self.s,
                                                self.random_state)
        self.optimizedModel = ugtm_gtm.optimize(X,
                                                self.initialModel,
                                                self.regul,
                                                self.niter,
                                                verbose=self.verbose)

        # Check fit
        check_is_fitted(self, ['optimizedModel'])

        # Input validation
        X = check_array(X)

        # Project new data onto fitted GTM
        self.projected = ugtm_gtm.projection(self.optimizedModel, X)

        # Output
        dic = {}
        dic["complete"] = self.projected
        dic["means"] = self.projected.matMeans
        dic["modes"] = self.projected.matModes
        dic["responsibilities"] = self.projected.matR
        return dic[self.model]

    def inverse_transform(self, matR):
        """Inverse transformation of responsibility onto the original data space

        Parameters
        ==========
        matR : array of shape (n_samples, n_nodes)

        Returns
        =======
        matY : array of shape (n_samples, n_dimensions)
        """
        weightedPhi = np.dot(matR, self.initialModel.matPhiMPlusOne)
        return np.dot(weightedPhi, self.optimizedModel.matW.T)


class eGTC(BaseEstimator, ClassifierMixin):
    """eGTC : GTC Bayesian classifier for sklearn pipelines.

    Arguments
    =========
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        m is the qrt of the number of RBF centers.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
    s : float, optional (default = 0.3)
        RBF width factor.
        One of four GTM hyperparameters (k, m, s, regul).
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = 0.1)
        One of four GTM hyperparameters (k, m, s, regul).
        Regularization coefficient.
    random_state : int (default = 1234)
        Random state.
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    verbose : bool, optional (default = False)
        Verbose mode (outputs loglikelihood values during EM algorithm).
    prior : {'estimated', 'equiprobable'}
        Type of prior for class map. Use 'estimated' to account for
        class imbalance.
    """

    def __init__(self, k=16, m=4, s=0.3, regul=0.1,
                 random_state=1234,
                 niter=200, verbose=False,
                 prior='estimated'):
        """Constructor for eGTC.

        Parameters
        ==========
        k : int, optional (default = 16)
            If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
            k is the sqrt of the number of GTM nodes.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
        m : int, optional (default = 4)
            If m is set to 0, m is computed as sqrt(k).
            m is the qrt of the number of RBF centers.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
        s : float, optional (default = 0.3)
            RBF width factor.
            One of four GTM hyperparameters (k, m, s, regul).
            Parameter to tune width of RBF functions.
            Impacts manifold flexibility.
        regul : float, optional (default = 0.1)
            One of four GTM hyperparameters (k, m, s, regul).
            Regularization coefficient.
        random_state : int (default = 1234)
            Random state.
        niter : int, optional (default = 200)
            Number of iterations for EM algorithm.
        verbose : bool, optional (default = False)
            Verbose mode (outputs loglikelihood values during EM algorithm).
        prior : {'estimated', 'equiprobable'}
            Type of prior for class map. Use 'estimated' to account for
            class imbalance.
        """
        self.k = k
        self.m = m
        self.s = s
        self.regul = regul
        self.random_state = random_state
        self.niter = niter
        self.verbose = verbose
        self.prior = prior

    def fit(self, X, y):
        """Constructs activity model f(X,y) using :func:`~ugtm.ugtm_landscape.classMap`.

        Parameters
        ==========

        X : array of shape (n_instances, n_dimensions)
            Data matrix.
        y : array of shape (n_instances,)
            Data labels.

        """
        X, y = check_X_y(X, y)

        self.initialModel = ugtm_gtm.initialize(X,
                                                self.k, self.m,
                                                self.s, self.random_state)
        self.optimizedModel = ugtm_gtm.optimize(X,
                                                self.initialModel,
                                                self.regul,
                                                self.niter,
                                                verbose=self.verbose)

        # compute activity model, posterior probabilities of class membership
        classmap = ugtm_landscape.classMap(
            self.optimizedModel, y, self.prior)
        self.node_probabilities = classmap.nodeClassP
        self.node_label = classmap.activityModel
        self.classes_ = unique_labels(y)

        # Return the classifier
        return self

    def predict(self, X):
        """Predicts new labels for X using :func:`~ugtm.ugtm_gtm.projection`.

        Parameters
        ==========

        X : array of shape (n_instances, n_dimensions)
            Data matrix.
        """
        # Check fit
        check_is_fitted(self, ['optimizedModel', 'node_probabilities'])

        # Input validation
        X = check_array(X)

        # Project new data onto fitted GTM
        projected = ugtm_gtm.projection(self.optimizedModel, X).matR

        # Dot product between projections and class probabilities
        self.posteriors = np.dot(projected, self.node_probabilities)
        self.predicted = np.argmax(self.posteriors, axis=1)
        return self.predicted


class eGTR(BaseEstimator, RegressorMixin):
    """eGTR: GTM nearest node(s) regressor for sklearn pipelines.

    Parameters
    ==========
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        m is the qrt of the number of RBF centers.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
    s : float, optional (default = 0.3)
        RBF width factor.
        One of four GTM hyperparameters (k, m, s, regul).
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = 0.1)
        One of four GTM hyperparameters (k, m, s, regul).
        Regularization coefficient.
    random_state : int (default = 1234)
        Random state.
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    verbose : bool, optional (default = False)
        Verbose mode (outputs loglikelihood values during EM algorithm).
    prior : {'estimated', 'equiprobable'}
        Type of prior for class map. Use 'estimated' to account for
        class imbalance.
    n_neighbors : int, optional (default = 2)
        Number of neighbors for kNN algorithm.
    representation : {'modes', 'means'}, optional
        Type of 2D representation used in kNN algorithm.
    """

    def __init__(self, k=16, m=4, s=0.3, regul=0.1,
                 random_state=1234,
                 niter=200, verbose=False,
                 n_neighbors=2, representation="modes"):
        """Constructor for eGTR.

        Parameters
        ==========
        k : int, optional (default = 16)
            If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
            k is the sqrt of the number of GTM nodes.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
        m : int, optional (default = 4)
            If m is set to 0, m is computed as sqrt(k).
            m is the qrt of the number of RBF centers.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
        s : float, optional (default = 0.3)
            RBF width factor.
            One of four GTM hyperparameters (k, m, s, regul).
            Parameter to tune width of RBF functions.
            Impacts manifold flexibility.
        regul : float, optional (default = 0.1)
            One of four GTM hyperparameters (k, m, s, regul).
            Regularization coefficient.
        random_state : int (default = 1234)
            Random state.
        niter : int, optional (default = 200)
            Number of iterations for EM algorithm.
        verbose : bool, optional (default = False)
            Verbose mode (outputs loglikelihood values during EM algorithm).
        n_neighbors : int, optional (default = 2)
            Number of neighbors for kNN algorithm.
        representation : {'modes', 'means'}, optional
            Type of 2D representation used in kNN algorithm.
        """
        self.k = k
        self.m = m
        self.s = s
        self.regul = regul
        self.random_state = random_state
        self.niter = niter
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.representation = representation

    def fit(self, X, y):
        """Constructs activity model f(X,y) using :func:`~ugtm.ugtm_landscape.landscape`.

        Parameters
        ==========

        X : array of shape (n_instances, n_dimensions)
            Data matrix.
        y : array of shape (n_instances,)
            Data labels.
        """
        X, y = check_X_y(X, y)

        # Train GTM

        self.initialModel = ugtm_gtm.initialize(X, self.k,
                                                self.m, self.s,
                                                self.random_state)
        self.optimizedModel = ugtm_gtm.optimize(X, self.initialModel,
                                                self.regul,
                                                self.niter,
                                                verbose=self.verbose)

        # Compute activity model = activity landscape
        self.node_label = ugtm_landscape.landscape(self.optimizedModel, y)

        # Return the regressor
        return self

    def predict(self, X):
        """Predicts new labels for X using :func:`~ugtm.ugtm_gtm.projection`.

        Parameters
        ==========

        X : array of shape (n_instances, n_dimensions)
            Data matrix.

        """
        # Check fit
        check_is_fitted(self, ['optimizedModel', 'node_label'])

        # Input validation
        X = check_array(X)

        # Project new data onto fitted GTM
        projected = ugtm_gtm.projection(self.optimizedModel, X)

        # Initialize knn model
        neighborModel = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric='euclidean')

        # Choose 2D GTM representation
        if self.representation == 'means':
            rep = projected.matMeans
        elif self.representation == 'modes':
            rep = projected.matModes

        # Initialize kNN model using nodes coordinates
        fitted = neighborModel.fit(self.optimizedModel.matX)

        # Compute distances between
        # test set projections and nodes on the map
        dist, nnID = fitted.kneighbors(rep, return_distance=True)
        dist[dist <= 0] = 10E-8  # np.finfo(float).tiny
        # The predicted value is the average of neareset landscape activities
        self.predicted = np.average(
            self.node_label[nnID], axis=1, weights=1 / ((dist)**2))

        # Return predictions
        return self.predicted


class eGTCnn(BaseEstimator, RegressorMixin):
    """eGTCnn: GTC nearest node classifier for sklearn pipelines.

    Arguments
    =========
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        m is the qrt of the number of RBF centers.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
    s : float, optional (default = 0.3)
        RBF width factor.
        One of four GTM hyperparameters (k, m, s, regul).
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = 0.1)
        One of four GTM hyperparameters (k, m, s, regul).
        Regularization coefficient.
    random_state : int (default = 1234)
        Random state.
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    verbose : bool, optional (default = False)
        Verbose mode (outputs loglikelihood values during EM algorithm).
    prior : {'estimated', 'equiprobable'}
        Type of prior for class map. Use 'estimated' to account for
        class imbalance.
    representation : {'modes', 'means'}, optional
        Type of 2D representation used in kNN algorithm.
    """

    def __init__(self, k=16, m=4, s=0.3, regul=0.1,
                 random_state=1234,
                 niter=200, verbose=False,
                 prior='estimated',
                 representation="modes"):
        """Constructor for eGTCnn.

        Parameters
        ==========
        k : int, optional (default = 16)
            If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
            k is the sqrt of the number of GTM nodes.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
        m : int, optional (default = 4)
            If m is set to 0, m is computed as sqrt(k).
            m is the qrt of the number of RBF centers.
            One of four GTM hyperparameters (k, m, s, regul).
            Ex: m = 5 means the RBF functions will be arranged on a 5x5 grid.
        s : float, optional (default = 0.3)
            RBF width factor.
            One of four GTM hyperparameters (k, m, s, regul).
            Parameter to tune width of RBF functions.
            Impacts manifold flexibility.
        regul : float, optional (default = 0.1)
            One of four GTM hyperparameters (k, m, s, regul).
            Regularization coefficient.
        random_state : int (default = 1234)
            Random state.
        niter : int, optional (default = 200)
            Number of iterations for EM algorithm.
        verbose : bool, optional (default = False)
            Verbose mode (outputs loglikelihood values during EM algorithm).
        prior : {'estimated', 'equiprobable'}
            Type of prior for class map. Use 'estimated' to account for
            class imbalance.
        representation : {'modes', 'means'}, optional
            Type of 2D representation used in kNN algorithm.
        """
        self.k = k
        self.m = m
        self.s = s
        self.regul = regul
        self.random_state = random_state
        self.niter = niter
        self.verbose = verbose
        self.n_neighbors = 1
        self.prior = prior
        self.representation = representation

    def fit(self, X, y):
        """Constructs activity model f(X,y) using :func:`~ugtm.ugtm_landscape.classMap`.

        Parameters
        ==========

        X : array of shape (n_instances, n_dimensions)
            Data matrix.
        y : array of shape (n_instances,)
            Data labels.

        """
        X, y = check_X_y(X, y)

        self.initialModel = ugtm_gtm.initialize(X, self.k,
                                                self.m, self.s,
                                                self.random_state)
        self.optimizedModel = ugtm_gtm.optimize(X, self.initialModel,
                                                self.regul,
                                                self.niter,
                                                verbose=self.verbose)

        # Compute activity model, posterior probabilities of class membership
        classmap = ugtm_landscape.classMap(
            self.optimizedModel, y, self.prior)
        self.node_probabilities = classmap.nodeClassP
        self.node_label = classmap.activityModel
        self.classes_ = unique_labels(y)

        # Return the classifier
        return self

    def predict(self, X):
        """Predicts new labels for X using :func:`~ugtm.ugtm_gtm.projection`.

        Parameters
        ==========

        X : array of shape (n_instances, n_dimensions)
            Data matrix.
        """

        # Check fit
        check_is_fitted(self, ['optimizedModel', 'node_label'])

        # Input validation
        X = check_array(X)

        # Project new data onto fitted GTM
        projected = ugtm_gtm.projection(self.optimizedModel, X)

        # Initialize knn model
        neighborModel = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric='euclidean')

        # Choose 2D GTM representation
        if self.representation == 'means':
            rep = projected.matMeans
        elif self.representation == 'modes':
            rep = projected.matModes

        # Initialize kNN model using nodes coordinates
        fitted = neighborModel.fit(self.optimizedModel.matX)

        # Compute distances between test set projections and nodes on the map
        nnID = fitted.kneighbors(rep, return_distance=False)

        # The predicted value is the label of the nearest node
        self.predicted = np.squeeze(self.node_label[nnID])

        # Return predictions
        return self.predicted.astype(int)
