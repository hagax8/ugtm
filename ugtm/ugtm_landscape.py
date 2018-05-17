from __future__ import print_function
import numpy as np


class ClassMap(object):
    def __init__(self, nodeClassP, nodeClassT, activityModel, uniqClasses):
        self.nodeClassP = nodeClassP
        self.nodeClassT = nodeClassT
        self.activityModel = activityModel
        self.uniqClasses = uniqClasses


def landscape(optimizedModel, activity):
    epsilon = 10e-8
    sums = np.sum(optimizedModel.matR+epsilon, axis=0)
    landscape = np.dot(activity.flatten(),
                       optimizedModel.matR+epsilon) / sums[None, :]
    return np.asarray(landscape)[0, :]


def classMap(optimizedModel, activity, prior="equiprobable"):
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
