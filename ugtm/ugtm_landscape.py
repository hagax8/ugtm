"""Builds continuous GTM class maps or landscapes using labels or activities.
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np


class ClassMap(object):
    r"""Class for ClassMap: Bayesian classification model for each GTM node.

    Arguments
    =========
    nodeClassT : array of shape (n_nodes, n_classes)
        Likelihood of each node :math:`k`
        given class :math:`C_i`:
        :math:`P(k|C_i) = \frac{\sum_{i_{c}}R_{i_{c},k}}{N_c}`.
    nodeClassP : array of shape (n_nodes, n_classes)
        Posterior probabilities of each class :math:`C_i`
        for each node :math:`k`:
        :math:`P(C_i|k) =\frac{P(k|C_i)P(C_i)}{\sum_i P(k|C_i)P(C_i)}`
    activityModel : array of shape (n_nodes,1)
        Class label attributed to each GTM node on the GTM node grid.
        Computed using argmax of posterior probabilities.
    uniqClasses : array of shape (n_classes,1)
        Unique class labels.
    """

    def __init__(self, nodeClassP, nodeClassT, activityModel, uniqClasses):
        r"""Constructor of ClassMap.

        Parameters
        ==========
        nodeClassT : array of shape (n_nodes, n_classes)
            Likelihood of each node :math:`k`
            given class :math:`C_i`:
            :math:`P(k|C_i) = \frac{\sum_{i_{c}}R_{i_{c},k}}{N_c}`.
        nodeClassP : array of shape (n_nodes, n_classes)
            Posterior probabilities of each class
            :math:`C_i` for each node :math:`k`:
            :math:`P(C_i|k) =\frac{P(k|C_i)P(C_i)}{\sum_i P(k|C_i)P(C_i)}`
        activityModel : array of shape (n_nodes,1)
            Class label attributed to each GTM node on the GTM node grid.
            Computed using argmax of posterior probabilities.
        uniqClasses : array of shape (n_classes,1)
            Unique class labels.
        """
        self.nodeClassP = nodeClassP
        self.nodeClassT = nodeClassT
        self.activityModel = activityModel
        self.uniqClasses = uniqClasses


def landscape(optimizedModel, activity):
    r"""Computes GTM landscapes based on activities (= continuous labels).

    Parameters
    ==========
    optimizedModel: an instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        The optimized GTM model.
    activity: array of shape (n_individuals,1)
        Activity vector (continuous labels) associated with the data
        used to compute the optimized GTM model.

    Returns
    =======
    array of shape (n_nodes,1)
        Activity landscape: associates each GTM node :math:`k`
        on the GTM node grid
        with an activity value, which is computed as an average mean of data
        activity values (continuous labels).
        If a = activities,
        r_k = vector of optimized GTM responsibilities for node k,
        and N = n_individuals:
        :math:`landscape_k = \frac{\mathbf{a \cdot r}_k}{\sum_i^{N}r_{ik}}`
    """
    epsilon = 10e-8
    sums = np.sum(optimizedModel.matR+epsilon, axis=0)
    landscape = np.dot(activity.flatten(),
                       optimizedModel.matR+epsilon) / sums[None, :]
    return np.asarray(landscape)[0, :]


def classMap(optimizedModel, activity, prior="estimated"):
    r"""Computes GTM class map based on discrete activities (= discrete labels)

    Parameters
    ==========
    optimizedModel: an instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`
        The optimized GTM model.
    activity: array of shape (n_individuals,1)
        Activity vector (discrete labels) associated with the data
        used to compute the optimized GTM model.
    prior: {estimated, equiprobable}, optional
        Type of prior used for Bayesian classifier.
        "equiprobable" assigns the same weight to all classes:
        :math:`P(C_i)=1/N_{classes}`.
        "estimated" accounts for class imbalance using
        the number of individuals in each class :math:`N(C_i)`:
        :math:`P(C_i)=N_{C_i}/N_{total}`

    Returns
    =======
    instance of :class:`~ugtm.ugtm_landscape.ClassMap`
        Computes a GTM bayesian model and returns an instance of
        :class:`~ugtm.ugtm_landscape.ClassMap`.

    Notes
    =====
    This function computes the likelihood of each GTM node given a class,
    the posterior probabilities of each class (using Bayes' theorem),
    and the class attributed to each node:

        1. output.nodeClassT:
           likelihood of each node :math:`k`
           given class :math:`C_i`:
           :math:`P(k|C_i) = \frac{\sum_{i_{c}}R_{i_{c},k}}{N_c}`.
        2. output.nodeClassP:
           posterior probabilities of each class
           :math:`C_i` for each node :math:`k`,
           using piors :math:`P(C_i)`:
           :math:`P(C_i|k) =\frac{P(k|C_i)P(C_i)}{\sum_i P(k|C_i)P(C_i)}`
        3. output.activityModel:
            Class label attributed to each GTM node on the GTM node grid.
            Computed using argmax of posterior probabilities.

    """
    uniqClasses, classVector = np.unique(activity, return_inverse=True)
    nClasses = uniqClasses.shape[0]
    n_nodes = optimizedModel.matR.shape[1]
    # posterior distribution
    nodeClassP = np.zeros([n_nodes, nClasses])
    # likelihood
    nodeClassT = np.zeros([n_nodes, nClasses])
    sumClass = np.zeros([nClasses])
    summe = np.zeros([n_nodes])
    for i in range(nClasses):
        sumClass[i] = (classVector == i).sum()
    if prior == "estimated":
        priors = sumClass/sumClass.sum()
    elif prior == "equiprobable":
        priors = np.zeros([nClasses])+(1.0/nClasses)

    for i in range(nClasses):
        for k in range(n_nodes):
            nodeClassT[k, i] = optimizedModel.matR[classVector ==
                                                   i, k].sum()/sumClass[i]

    for i in range(nClasses):
        for k in range(n_nodes):
            nodeClassP[k, i] = nodeClassT[k, i]*priors[i]
            summe[k] += nodeClassP[k, i]

    for i in range(nClasses):
        for k in range(n_nodes):
            if summe[k] != 0.0:
                nodeClassP[k, i] = nodeClassP[k, i]/summe[k]

    for k in range(n_nodes):
        if summe[k] == 0.0:
            for i in range(nClasses):
                nodeClassP[k, i] = 1/nClasses

    nodeClass = np.argmax(nodeClassP, axis=1)
    return(ClassMap(nodeClassP, nodeClassT, nodeClass, uniqClasses))
