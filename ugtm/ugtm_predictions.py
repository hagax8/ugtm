from __future__ import print_function
import numpy as np
import math
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .ugtm_gtm import initialize
from .ugtm_gtm import optimize
from .ugtm_gtm import projection
from . import ugtm_landscape
from . import ugtm_preprocess


def predictNN(optimizedModel, labels, new_data, modeltype, n_neighbors=1,
              representation="modes", prior="equiprobable"):
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
    return predicted


def predictNNSimple(train, test, labels, n_neighbors=1,
                    modeltype='regression'):
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
    return predicted


def predictBayes(optimizedModel, labels, new_data, prior="equiprobable"):
    activityModel = ugtm_landscape.classMap(optimizedModel,
                                            labels, prior).nodeClassP
    projected = projection(optimizedModel, new_data).matR
    predicted = np.argmax(np.matmul(projected, activityModel), axis=1)
    return predicted


def advancedGTC(train, labels, test, n_neighbors=1, representation="modes",
                niter=200, k=0, m=0, doPCA=False,
                n_components=-1, missing=False,
                missing_strategy='most_frequent', random_state=1234,
                predict_mode="bayes", prior="equiprobable", l=0.1, s=0.3):
    if k <= 0:
        k = int(math.sqrt(5*math.sqrt(train.shape[0])))+2
    if m <= 0:
        m = int(math.sqrt(k))
    if n_components == -1 and doPCA:
        pca = PCA(random_state=random_state)
        pca.fit(train)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used n_components explaining 80%% of the variance = %s\n"
              % n_components)
    if l < 0.0:
        l = 0.1
    if s <= 0.0:
        s = 0.3
    processed = ugtm_preprocess.processTrainTest(train, test, doPCA,
                                                 n_components, missing,
                                                 missing_strategy)
    initialModel = initialize(processed.train, k, m,
                              s, random_state=random_state)
    optimizedModel = optimize(processed.train, initialModel, l, niter, 0)
    prediction = advancedPredictBayes(
        optimizedModel, labels, processed.test, prior)
    return prediction


def advancedPredictBayes(optimizedModel, labels,
                         new_data, prior="equiprobable"):
    predicted = {}
    cl = ugtm_landscape.classMap(optimizedModel, labels, prior)
    activityModel = cl.nodeClassP
    projected = projection(optimizedModel, new_data)
    predicted["optimizedModel"] = optimizedModel
    predicted["indiv_projections"] = projected
    predicted["indiv_probabilities"] = np.matmul(projected.matR, activityModel)
    predicted["indiv_predictions"] = np.argmax(
        predicted["indiv_probabilities"], axis=1)
    predicted["group_projections"] = np.mean(projected.matR, axis=0)
    predicted["group_probabilities"] = np.matmul(
        predicted["group_projections"], activityModel)
    predicted["uniqClasses"] = cl.uniqClasses
    return predicted


def printClassPredictions(prediction, output):
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


def GTC(train, labels, test, k=0, m=0, s=0.3, l=0.1, n_neighbors=1, niter=200,
        representation="modes", doPCA=False, n_components=-1, missing=False,
        missing_strategy='most_frequent', random_state=1234,
        predict_mode="bayes", prior="equiprobable"):
    if k == 0:
        k = int(math.sqrt(5*math.sqrt(train.shape[0])))+2
    if m == 0:
        m = int(math.sqrt(k))
    processed = ugtm_preprocess.processTrainTest(train, test, doPCA,
                                                 n_components, missing,
                                                 missing_strategy)
    initialModel = initialize(processed.train, k, m,
                              s, random_state=random_state)
    optimizedModel = optimize(processed.train, initialModel, l, niter, 0)
    if predict_mode == "knn":
        prediction = predictNN(optimizedModel, labels, processed.test,
                               "classification", n_neighbors,
                               representation, prior)
    elif predict_mode == "bayes":
        prediction = predictBayes(
            optimizedModel, labels, processed.test, prior)
    return prediction


def GTR(train, labels, test, k=0, m=0, s=0.3, l=0.1, n_neighbors=1, niter=200,
        representation="modes", doPCA=False, n_components=-1,
        missing=False, missing_strategy='most_frequent', random_state=1234):
    if k == 0:
        k = int(math.sqrt(5*math.sqrt(train.shape[0])))+2
    if m == 0:
        m = int(math.sqrt(k))
    processed = ugtm_preprocess.processTrainTest(train, test,
                                                 doPCA, n_components)
    initialModel = initialize(processed.train, k, m,
                              s, random_state=random_state)
    optimizedModel = optimize(processed.train, initialModel, l, niter, 0)
    prediction = predictNN(optimizedModel=optimizedModel, labels=labels,
                           new_data=processed.test, modeltype="regression",
                           n_neighbors=n_neighbors,
                           representation=representation)
    return prediction
