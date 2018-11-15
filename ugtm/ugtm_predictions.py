"""GTC (GTM classification) and GTR (GTM regression)
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .ugtm_gtm import initialize
from .ugtm_gtm import optimize
from .ugtm_gtm import projection
from . import ugtm_landscape
from . import ugtm_preprocess


def predictNN(optimizedModel, labels, new_data, modeltype="regression",
              n_neighbors=1,
              representation="modes", prior="estimated"):
    r"""GTM nearest node(s) classification or regression.

    Parameters
    ==========
    optimizedModel : instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized GTM model built using a training set
        of shape (n_individuals, n_dimensions)
    labels : array of shape (n_individuals, 1)
        Labels (discrete or continuous) associated with training set
    new_data : array of shape (n_test, n_dimensions)
        New data matrix (test set).
    modeltype : {'classification', 'regression'}, optional
        Choice between classification and regression.
    n_neighbors : int, optional (default = 1)
        Number of nodes to take into account in kNN algorithm.
        NB: for classification, n_neighbors is always equal to 1.
    representation : {'modes', 'means'}, optional
        Defines GTM representation type: mean or mode of responsibilities.
    prior : {'estimated', 'equiprobable'}, optional
        Only used for classification.
        Sets priors (Bayes' theorem) in :func:`~ugtm.ugtm_landscape.classMap`.

    Returns
    =======
    array of shape (n_test, 1)
        Predicted outcome.

    Notes
    =====
    This function implements classification or regression based
    on nearest GTM nodes:

        1. If (modeltype == 'classification'), generate GTM class map
           (:func:`~ugtm.ugtm_landscape.classMap`);
           if (modeltype == 'regression'), generate GTM landscape
           (:func:`~ugtm.ugtm_landscape.landscape`)
        2. Project new data (:func:`~ugtm.ugtm_gtm.projection`)
           on optimized GTM model (:class:`~ugtm.ugtm_classes.OptimizedGTM`)
        3. Depending on provided parameters,
           choose means or modes as GTM coordinates for the new data
        4. Find the nodes closest to the new data GTM coordinates
           (sklearn function kneighbors)
        5. Retrieve predicted outcomes corresponding to nodes on class map
           (classification task) or landscape (regression task)
        6. If (modeltype == 'classification'), the predicted outcome is the
           outcome of the nearest node on the class map;
           if (modeltype == 'regression'),
           the predicted outcome is the average outcome of the k nearest nodes
           (k = n_neighbors), weighted by inverse squared distances
           (weights=1/((dist)**2))

    """
    if modeltype == 'regression':
        activityModel = ugtm_landscape.landscape(optimizedModel, labels)
    elif modeltype == 'classification':
        n_neighbors = 1
        activityModel = ugtm_landscape.classMap(optimizedModel,
                                                labels, prior).activityModel
    projected = projection(optimizedModel, new_data)
    neighborModel = NearestNeighbors(
        n_neighbors=n_neighbors, metric='euclidean')
    fitted = neighborModel.fit(optimizedModel.matX)
    if representation == 'means':
        rep = projected.matMeans
    elif representation == 'modes':
        rep = projected.matModes
    if modeltype == 'regression' and n_neighbors > 1:
        dist, nnID = fitted.kneighbors(rep, return_distance=True)
        dist[dist <= 0] = np.finfo(float).tiny
        predicted = np.average(
            activityModel[nnID], axis=1, weights=1/((dist)**2))
    else:
        nnID = fitted.kneighbors(rep, return_distance=False)
        predicted = activityModel[nnID]
    return np.squeeze(predicted).astype(int)


def predictNNSimple(train, test, labels, n_neighbors=1,
                    modeltype='regression'):
    r"""Nearest neighbor(s) classification or regression.

    Parameters
    ==========
    train : array of shape (n_train, n_dimensions)
        Train set data matrix.
    test : array of shape (n_test, n_dimensions)
        Test set data matrix.
    labels : array of shape (n_train, 1)
        Labels (discrete or continuous) for the training set.
    n_neighbors : int, optional (default = 1)
        Number of nodes to take into account in kNN algorithm.
    modeltype : {'classification', 'regression'}, optional
        Choice between classification and regression.

    Returns
    =======
    array of shape (n_test, 1)
        Predicted outcome.

    Notes
    =====
    This function implements classification or regression based
    on classical kNN algorithm.
    """
    if modeltype == 'regression' and n_neighbors > 1:
        neighborModel = NearestNeighbors(
            n_neighbors=n_neighbors, metric='euclidean')
        fitted = neighborModel.fit(train)
        dist, nnID = fitted.kneighbors(test, return_distance=True)
        dist[dist <= 0] = np.finfo(float).tiny
        predicted = np.average(labels[nnID], axis=1, weights=1/((dist)**2))

    else:
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(train, labels)
        predicted = clf.predict(test)
    return np.squeeze(predicted).astype(int)


def predictBayes(optimizedModel, labels, new_data, prior="estimated"):
    r""" Bayesian GTM classifier (GTC Bayes).

    Parameters
    ==========
    optimizedModel : instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized GTM model built using a training set
        of shape (n_individuals, n_dimensions)
    labels : array of shape (n_individuals, 1)
        Labels (discrete or continuous) associated with training set
    new_data : array of shape (n_test, n_dimensions)
        New data matrix (test set).
    prior : {'estimated', 'equiprobable'}, optional
        Only used for classification.
        Sets priors (Bayes' theorem) in :func:`~ugtm.ugtm_landscape.classMap`.

    Returns
    =======
    array of shape (n_test, 1)
        Predicted outcome.

    Notes
    =====
    This function computes GTM class predictions by using
    posterior probabilities of classes weighted by responsibilities.
    Similar to maximum a posterior (MAP) estimator.

        1. generate GTM class map
           (:func:`~ugtm.ugtm_landscape.classMap`);
        2. Project new data (:func:`~ugtm.ugtm_gtm.projection`)
           on optimized GTM model (:class:`~ugtm.ugtm_classes.OptimizedGTM`)
        3. Projected data responsibilities R are used as weights
           to find outcome :math:`C_{max}` for each tested instance:
           :math:`C_{max} = \operatorname*{arg\,max}_C \sum_k{R_{ki} P(C|k)}`

    """
    activityModel = ugtm_landscape.classMap(optimizedModel,
                                            labels, prior).nodeClassP
    projected = projection(optimizedModel, new_data).matR
    predicted = np.argmax(np.dot(projected, activityModel), axis=1)
    return predicted


def advancedGTC(train, labels, test, n_neighbors=1, representation="modes",
                niter=200, k=16, m=4, regul=0.1, s=0.3, doPCA=False,
                n_components=-1, missing=False,
                missing_strategy='median', random_state=1234,
                predict_mode="bayes", prior="estimated"):
    r"""Run GTC (GTM classification): advanced Bayes

    Parameters
    ==========
    train : array of shape (n_train, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_train, 1)
        Labels for train set.
    test : array of shape (n_test, n_dimensions)
        Test set data matrix.
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        (generally good rule of thumb).
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
        Impacts manifold flexibility.
    n_neighbors : int, optional (default = 1)
        Number of neighbors for kNN algorithm (number of nearest nodes).
        At the moment, n_neighbors is always equal to 1.
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    representation : {"modes", "means"}
        2D GTM representation for the test set, used for kNN algorithms:
        "modes" for position with max. responsibility,
        "means" for average position (usual GTM representation)
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    n_components : int, optional (default = -1)
        Number of components for PCA pre-processing.
        If set to -1, keep principal components
        accounting for 80% of data variance.
    missing : bool, optional (default = True)
        Replace missing values (calls scikit-learn functions).
    missing_strategy : str, optional (default = 'median')
        Scikit-learn missing data strategy.
    random_state : int, optional (default = 1234)
        Random state.
    predict_mode : {"bayes"}, optional
        At the moment, only the GTM Bayes classifier is available;
        ("bayes", output of :func:`~ugtm.ugtm_predictions.advancedPredictBayes`).
    prior : {"estimated", "equiprobable"}, optional
        Type of prior used to build GTM class map
        (:func:`~ugtm.ugtm_landscape.classMap`).
        Choose "estimated" to account for class imbalance.

    Returns
    =======
    a dict

        The output is a dictionary defined as follows:

            1. output["optimizedModel"]: original training set GTM model,
               instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
            2. output["indiv_projections"]: test set GTM model,
               instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
            3. output["indiv_probabilities"]: class probabilities
               for each individual (= dot product between test responsibility
               matrix and posterior class probabilities)
            4. output["indiv_predictions"]: class prediction for each
               individual (argmax of output["indiv_probabilities"])
            5. output["group_projections"]: average responsibility vector
               for the entire test set
            6. output["group_probabilities"]: posterior class probabilities
               for the entire test set (dot product between
               output["group_projections"] and posterior class probabilities)
            7. output["uniqClasses"]: classes

    Notes
    =====
    The GTM nearest node classifier (predict_mode = "knn",
    :func:`~ugtm.ugtm_predictions.predictNN`):

        1. A GTM class map (GTM colored by class)
           is built using the training set
           (:func:`~ugtm.ugtm_landscape.classMap`);
           the class map is discretized into nodes,
           and each node has a class label
        2. The test set is projected onto the GTM map
        3. A 2D GTM representation is chosen for the test set
           (representation = modes or means)
        4. Nearest node on the GTM map is found for each test set individual
        5. The predicted label for each individual is the label of its
           nearest node on the GTM map

    The GTM Bayes classifier (predict_mode = "bayes",
    :func:`~ugtm.ugtm_predictions.predictBayes`):

        1. A GTM class map (GTM colored by class)
           is built using the training set
           (:func:`~ugtm.ugtm_landscape.classMap`);
           the class map is discretized into nodes,
           and each node has posterior class probabilities
        2. The test set is projected onto the GTM map
        3. The GTM representation for each individual is its responsibility
           vector (posterior probability distribution on the map)
        4. The probabilities of belonging to each class for a
           specific individual are computed
           as an average of posterior class probabilities
           (array of shape (n_nodes_n,classes)), weighted by the individual's
           responsibilities on the GTM map (array of shape (1, n_nodes))

    """
    if k == 0:
        k = int(np.sqrt(5*np.sqrt(train.shape[0])))+2
    if m == 0:
        m = int(np.sqrt(k))
    if n_components == -1 and doPCA:
        pca = PCA(random_state=random_state)
        pca.fit(train)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used n_components explaining 80%% of the variance = %s\n"
              % n_components)
    if regul < 0.0:
        regul = 0.1
    if s <= 0.0:
        s = 0.3
    processed = ugtm_preprocess.processTrainTest(train, test, doPCA,
                                                 n_components, missing,
                                                 missing_strategy)
    initialModel = initialize(processed.train, k, m,
                              s, random_state=random_state)
    optimizedModel = optimize(processed.train, initialModel, regul, niter, 0)
    prediction = advancedPredictBayes(
        optimizedModel, labels, processed.test, prior)
    return prediction


def advancedPredictBayes(optimizedModel, labels,
                         new_data, prior="estimated"):
    r"""Bayesian GTM classifier: complete model

    Parameters
    ==========
    optimizedModel : instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        Optimized GTM model built using a training set
        of shape (n_individuals, n_dimensions)
    labels : array of shape (n_individuals, 1)
        Labels (discrete or continuous) associated with training set
    new_data : array of shape (n_test, n_dimensions)
        New data matrix (test set).
    prior : {'estimated', 'equiprobable'}, optional
        Only used for classification.
        Sets priors (Bayes' theorem) in :func:`~ugtm.ugtm_landscape.classMap`.

    Returns
    =======
    a dict

        The output is a dictionary defined as follows:

            1. output["optimizedModel"]: original training set GTM model,
               instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
            2. output["indiv_projections"]: test set GTM model,
               instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
            3. output["indiv_probabilities"]: class probabilities
               for each individual (= dot product between test responsibility
               matrix and posterior class probabilities)
            4. output["indiv_predictions"]: class prediction for each
               individual (argmax of output["indiv_probabilities"])
            5. output["group_projections"]: average responsibility vector
               for the entire test set
            6. output["group_probabilities"]: posterior class probabilities
               for the entire test set (dot product between
               output["group_projections"] and posterior class probabilities)
            7. output["uniqClasses"]: classes

    Notes
    =====
    This function computes GTM class predictions by using
    posterior probabilities of classes weighted by responsibilities.

        1. generate GTM class map
           (:func:`~ugtm.ugtm_landscape.classMap`);
        2. Project new data (:func:`~ugtm.ugtm_gtm.projection`)
           on optimized GTM model (:class:`~ugtm.ugtm_classes.OptimizedGTM`)
        3. Projected data responsibilities R are used as weights
           to find outcome :math:`C_{max}` for each tested instance:
           :math:`C_{max} = \operatorname*{arg\,max}_C \sum_k{R_{ki} P(C|k)}`

    The algorithm is the same as
    in :func:`~ugtm.ugtm_predicions.predictBayes`, but this function
    returns a complete
    output including original training set optimized GTM model, test set
    GTM model, individual class probabilities for each individual,
    class prediction for each individual, group projections (average position
    of the whole test set on the map), class probabilities for the whole test
    set, and classes used to build the classification model.
    """
    predicted = {}
    cl = ugtm_landscape.classMap(optimizedModel, labels, prior)
    activityModel = cl.nodeClassP
    projected = projection(optimizedModel, new_data)
    predicted["optimizedModel"] = optimizedModel
    predicted["indiv_projections"] = projected
    predicted["indiv_probabilities"] = np.dot(projected.matR, activityModel)
    predicted["indiv_predictions"] = np.argmax(
        predicted["indiv_probabilities"], axis=1)
    predicted["group_projections"] = np.mean(projected.matR, axis=0)
    predicted["group_probabilities"] = np.dot(
        predicted["group_projections"], activityModel)
    predicted["uniqClasses"] = cl.uniqClasses
    return predicted


def printClassPredictions(prediction, output):
    r"""Print output of :func:`~ugtm.ugtm_predictions.advancedPredictBayes`.

    Parameters
    ==========
    prediction : dict
        Output of :func:`~ugtm.ugtm_predictions.advancedPredictBayes`.
        With following keys:
        "optimizedModel": :class:`~ugtm.ugtm_classes.OptimizedGTM`,
        "indiv_projections": :class:`~ugtm.ugtm_classes.OptimizedGTM`,
        "indiv_probabilities": array of shape (n_individuals, n_classes),
        "indiv_predictions": array of shape (n_individuals, 1),
        "group_projections": array of shape (n_nodes, 1),
        "group_probabilities": array of shape (n_probabilities, 1),
        "uniqClasses": array of shape(n_classes, 1)
    output : str
        Output path to write class prediction model (prediction dictionary).

    Returns
    =======
    CSV files

        1. output_indiv_probabilities.csv
        2. output_indiv_predictions.csv
        3. output_group_probabilities.csv

    """
    string = "Classes_in_this_order:"
    count = 0
    grouproba = prediction["group_probabilities"]
    for i in range(len(prediction["uniqClasses"])):
        string += str(count)+"="+str(prediction["uniqClasses"][i])+";"
        count = count + 1
    predvec = [prediction["uniqClasses"][j]
               for j in prediction["indiv_predictions"]]
    np.savetxt(fname=output+"_indiv_probabilities.csv",
               X=prediction["indiv_probabilities"],
               delimiter=",", header=string, fmt='%.2f')
    np.savetxt(fname=output+"_indiv_predictions.csv",
               X=prediction["indiv_predictions"],
               delimiter=",", header=string, fmt='%s')
    np.savetxt(fname=output+"_indiv_predictions_label.csv",
               X=predvec, delimiter=",",
               header=string, fmt='%s')
    np.savetxt(fname=output+"_group_probabilities.csv",
               X=grouproba.reshape(1, grouproba.shape[0]),
               delimiter=",", header=string, fmt='%.2f')
    print("Wrote to disk:")
    print("%s: individual probabilities" % (output+"_indiv_probabilities.csv"))
    print("%s: individual predictions" % (output+"_indiv_predictions.csv"))
    print("%s: group probabilities" % (output+"_group_probabilities.csv"))
    print("")


def GTC(train, labels, test, k=16, m=4, s=0.3, regul=0.1, n_neighbors=1, niter=200,
        representation="modes", doPCA=False, n_components=-1, missing=False,
        missing_strategy='median', random_state=1234,
        predict_mode="bayes", prior="estimated"):
    r"""Run GTC (GTM classification): Bayes or nearest node algorithm.

    Parameters
    ==========
    train : array of shape (n_train, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_train, 1)
        Labels for train set.
    test : array of shape (n_test, n_dimensions)
        Test set data matrix.
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        (generally good rule of thumb).
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
        Impacts manifold flexibility.
    n_neighbors : int, optional (default = 1)
        Number of neighbors for kNN algorithm (number of nearest nodes).
        At the moment, n_neighbors is always equal to 1.
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    representation : {"modes", "means"}
        2D GTM representation for the test set, used for kNN algorithms:
        "modes" for position with max. responsibility,
        "means" for average position (usual GTM representation)
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    n_components : int, optional (default = -1)
        Number of components for PCA pre-processing.
        If set to -1, keep principal components
        accounting for 80% of data variance.
    missing : bool, optional (default = True)
        Replace missing values (calls scikit-learn functions).
    missing_strategy : str, optional (default = 'median')
        Scikit-learn missing data strategy.
    random_state : int, optional (default = 1234)
        Random state.
    predict_mode : {"bayes", "knn"}, optional
        Choose between nearest node algorithm
        ("knn", output of :func:`~ugtm.ugtm_predictions.predictNN`)
        or GTM Bayes classifier
        ("bayes", output of :func:`~ugtm.ugtm_predictions.predictBayes`).
        NB: the kNN algorithm is limited to only 1 nearest node at the moment
        (n_neighbors = 1).
    prior : {"estimated", "equiprobable"}, optional
        Type of prior used to build GTM class map
        (:func:`~ugtm.ugtm_landscape.classMap`).
        Choose "estimated" to account for class imbalance.

    Returns
    =======
    array of shape (n_test, 1)
        Predicted class for test set individuals.

    Notes
    =====
    The GTM nearest node classifier (predict_mode = "knn",
    :func:`~ugtm.ugtm_predictions.predictNN`):

        1. A GTM class map (GTM colored by class)
           is built using the training set
           (:func:`~ugtm.ugtm_landscape.classMap`);
           the class map is discretized into nodes,
           and each node has a class label
        2. The test set is projected onto the GTM map
        3. A 2D GTM representation is chosen for the test set
           (representation = modes or means)
        4. Nearest node on the GTM map is found for each test set individual
        5. The predicted label for each individual is the label of its
           nearest node on the GTM map

    The GTM Bayes classifier (predict_mode = "bayes",
    :func:`~ugtm.ugtm_predictions.predictBayes`):

        1. A GTM class map (GTM colored by class)
           is built using the training set
           (:func:`~ugtm.ugtm_landscape.classMap`);
           the class map is discretized into nodes,
           and each node has posterior class probabilities
        2. The test set is projected onto the GTM map
        3. The GTM representation for each individual is its responsibility
           vector (posterior probability distribution on the map)
        4. The probabilities of belonging to each class for a
           specific individual are computed
           as an average of posterior class probabilities
           (array of shape (n_nodes_n,classes)), weighted by the individual's
           responsibilities on the GTM map (array of shape (1, n_nodes))

    """
    if k == 0:
        k = int(np.sqrt(5*np.sqrt(train.shape[0])))+2
    if m == 0:
        m = int(np.sqrt(k))
    processed = ugtm_preprocess.processTrainTest(train, test, doPCA,
                                                 n_components, missing,
                                                 missing_strategy)
    initialModel = initialize(processed.train, k, m,
                              s, random_state=random_state)
    optimizedModel = optimize(processed.train, initialModel, regul, niter, 0)
    if predict_mode == "knn":
        prediction = predictNN(optimizedModel, labels, processed.test,
                               "classification", n_neighbors,
                               representation, prior)
    elif predict_mode == "bayes":
        prediction = predictBayes(
            optimizedModel, labels, processed.test, prior)
    return prediction


def GTR(train, labels, test, k=16, m=4, s=0.3, regul=0.1, n_neighbors=1, niter=200,
        representation="modes", doPCA=False, n_components=-1,
        missing=False, missing_strategy='median', random_state=1234):
    r"""Run GTR (GTM nearest node(s) regression).

    Parameters
    ==========
    train : array of shape (n_train, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_train, 1)
        Labels for train set.
    test : array of shape (n_test, n_dimensions)
        Test set data matrix.
    k : int, optional (default = 16)
        If k is set to 0, k is computed as sqrt(5*sqrt(n_individuals))+2.
        k is the sqrt of the number of GTM nodes.
        One of four GTM hyperparameters (k, m, s, regul).
        Ex: k = 25 means the GTM will be discretized into a 25x25 grid.
    m : int, optional (default = 4)
        If m is set to 0, m is computed as sqrt(k).
        (generally good rule of thumb).
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
        Impacts manifold flexibility.
    n_neighbors : int, optional (default = 1)
        Number of neighbors for kNN algorithm (number of nearest nodes).
    niter : int, optional (default = 200)
        Number of iterations for EM algorithm.
    representation : {"modes", "means"}
        2D GTM representation for the test set, used for kNN algorithms:
        "modes" for position with max. responsibility,
        "means" for average position (usual GTM representation)
    doPCA : bool, optional (default = False)
        Apply PCA pre-processing.
    n_components : int, optional (default = -1)
        Number of components for PCA pre-processing.
        If set to -1, keep principal components
        accounting for 80% of data variance.
    missing : bool, optional (default = True)
        Replace missing values (calls scikit-learn functions).
    missing_strategy : str, optional (default = 'median')
        Scikit-learn missing data strategy.
    random_state : int, optional (default = 1234)
        Random state.

    Returns
    =======
    array of shape (n_test, 1)
        Predicted class for test set individuals.

    Notes
    =====
    The GTM nearest node(s) regression
    (:func:`~ugtm.ugtm_predictions.predictNN`):

        1. A GTM landscape (GTM colored by activity)
           is built using the training set
           (:func:`~ugtm.ugtm_landscape.landscape`);
           the landscape is discretized into nodes,
           and each node has an estimated activity value
        2. The test set is projected onto the GTM map
        3. A 2D GTM representation is chosen for the test set
           (representation = modes or means)
        4. Nearest node(s) on the GTM map is found for each test set individual
        5. The predicted activity for each individual is a weighted average
           of nearest node activities.

    """
    if k == 0:
        k = int(np.sqrt(5*np.sqrt(train.shape[0])))+2
    if m == 0:
        m = int(np.sqrt(k))
    processed = ugtm_preprocess.processTrainTest(train, test,
                                                 doPCA, n_components)
    initialModel = initialize(processed.train, k, m,
                              s, random_state=random_state)
    optimizedModel = optimize(processed.train, initialModel, regul, niter, 0)
    prediction = predictNN(optimizedModel=optimizedModel, labels=labels,
                           new_data=processed.test, modeltype="regression",
                           n_neighbors=n_neighbors,
                           representation=representation)
    return prediction
