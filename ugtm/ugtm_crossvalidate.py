"""Cross-validation support for GTC and GTR models (also SVM and PCA).
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.svm import SVR
import scipy.stats as st
from scipy.stats import t
from . import ugtm_predictions
from . import ugtm_preprocess


def crossvalidateGTC(data, labels, k=16, m=4, s=-1.0, regul=1.0,
                     n_neighbors=1, niter=200,
                     representation="modes",
                     doPCA=False, n_components=-1,
                     missing=False, missing_strategy='median',
                     random_state=1234, predict_mode="bayes",
                     prior="estimated",
                     n_folds=5, n_repetitions=10):
    r"""Cross-validate GTC model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_individuals, 1)
        Labels for train set.
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
    s : float, optional (default = -1)
        RBF width factor. Default (-1) is to try different values.
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = -1)
        Regularization coefficient. Default (-1) is to try different values.
        Impacts manifold flexibility.
    n_neighbors : int, optional (default = 1)
        Number of neighbors for kNN algorithm (number of nearest nodes).
        At the moment, n_neighbors for GTC is always equal to 1.
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
    n_folds : int, optional (default = 5)
        Number of CV folds.
    n_repetitions : int, optional (default = 10)
        Number of CV iterations.
    """
    print("")
    print("k = sqrt(grid size), m = sqrt(radial basis function grid size), "
          "regul = regularization, s = RBF width factor")
    print("")
    uniqClasses, labels = np.unique(labels, return_inverse=True)
    nClasses = len(uniqClasses)
    print("Classes: ", uniqClasses)
    print("nClasses: %s" % (nClasses))
    print("")
    print("model\tparameters=k:m:s:regul\t"
          "recall with CI\tprecision with CI\tF1-score with CI")
    print("")
    if k == 0:
        k = int(np.sqrt(5*np.sqrt(data.shape[0])))+2
    if m == 0:
        m = int(np.sqrt(k))
    if n_components == -1 and doPCA:
        pca = PCA(random_state=random_state)
        pca.fit(data)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used number of components explaining 80%% of "
              "the variance in whole data set = %s\n" %
              n_components)
    if regul < 0.0:
        lvec = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    else:
        lvec = [regul]
    if s <= 0.0:
        svec = [0.25, 0.5, 1.0, 1.50, 2.0]
    else:
        svec = [s]
    savemean = -9999
    nummodel = 0
    savemodel = ""
    for s in svec:
        for regul in lvec:
            modelstring = str(k)+':'+str(m)+":"+str(s)+":"+str(regul)
            nummodel += 1
            recallvec = []
            precisionvec = []
            f1vec = []
            recallclassvec = np.array([])
            precisionclassvec = np.array([])
            f1classvec = np.array([])
            meanclass = np.zeros(nClasses)
            meanprecisionclass = np.zeros(nClasses)
            meanf1class = np.zeros(nClasses)
            seclass = np.zeros(nClasses)
            seprecisionclass = np.zeros(nClasses)
            sef1class = np.zeros(nClasses)
            hclass = np.zeros(nClasses)
            hprecisionclass = np.zeros(nClasses)
            hf1class = np.zeros(nClasses)
            for j in range(n_repetitions):
                ss = KFold(n_splits=n_folds, shuffle=True, random_state=j)
                y_true = []
                y_pred = []
                for train_index, test_index in ss.split(data):
                    train = np.copy(data[train_index])
                    test = np.copy(data[test_index])
                    prediction = ugtm_predictions.GTC(train=train,
                                                      labels=labels[train_index],
                                                      test=test, k=k,
                                                      m=m, s=s, regul=regul,
                                                      n_neighbors=n_neighbors,
                                                      niter=niter,
                                                      representation=representation,
                                                      doPCA=doPCA,
                                                      n_components=n_components,
                                                      random_state=random_state,
                                                      missing=missing,
                                                      missing_strategy=missing_strategy,
                                                      predict_mode=predict_mode,
                                                      prior=prior)
                    y_true = np.append(y_true, labels[test_index])
                    y_pred = np.append(y_pred, prediction)
                recall = recall_score(y_true, y_pred, average='weighted')
                precision = precision_score(
                    y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
                recallvec = np.append(recallvec, recall)
                precisionvec = np.append(precisionvec, precision)
                f1vec = np.append(f1vec, f1)
                recallclass = recall_score(y_true, y_pred, average=None)
                precisionclass = precision_score(y_true, y_pred, average=None)
                f1class = f1_score(y_true, y_pred, average=None)
                if(j == 0):
                    recallclassvec = recallclass
                    precisionclassvec = precisionclass
                    f1classvec = f1class
                else:
                    recallclassvec = np.vstack([recallclassvec, recallclass])
                    precisionclassvec = np.vstack(
                        [precisionclassvec, precisionclass])
                    f1classvec = np.vstack([f1classvec, f1class])
            mean, se = np.mean(recallvec), st.sem(recallvec)
            meanprecision, seprecision = np.mean(
                precisionvec), st.sem(precisionvec)
            meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
            h = se * t._ppf((1+0.95)/2., len(recallvec)-1)
            hprecision = seprecision * \
                t._ppf((1+0.95)/2., len(precisionvec)-1)
            hf1 = sef1 * t._ppf((1+0.95)/2., len(f1vec)-1)
            if(meanf1 > savemean):
                savemean = meanf1
                savemodel = "Model "+str(nummodel)
            for i in range(0, nClasses):
                meanclass[i] = np.mean(recallclassvec[:, i])
                seclass[i] = st.sem(recallclassvec[:, i])
                meanf1class[i] = np.mean(f1classvec[:, i])
                sef1class[i] = st.sem(f1classvec[:, i])
                meanprecisionclass[i] = np.mean(precisionclassvec[:, i])
                seprecisionclass[i] = st.sem(precisionclassvec[:, i])
                hclass[i] = seclass[i] * \
                    t._ppf((1+0.95)/2., len(recallclassvec[:, i])-1)
                hprecisionclass[i] = seprecisionclass[i] \
                    * t._ppf((1+0.95)/2., len(precisionclassvec[:, i])-1)
                hf1class[i] = sef1class[i] * \
                    t._ppf((1+0.95)/2., len(f1classvec[:, i])-1)
            print("Model %s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
                  % (nummodel, modelstring, mean, h,
                     meanprecision, hprecision, meanf1, hf1))
            for i in range(nClasses):
                print("Class=%s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
                      % (uniqClasses[i], modelstring, meanclass[i],
                         hclass[i], meanprecisionclass[i],
                         hprecisionclass[i], meanf1class[i], hf1class[i]))
            print('')

    print('')
    print("########best GTC model##########")
    print(savemodel)
    print("")


def crossvalidateGTR(data, labels, k=16, m=4, s=-1, regul=-1,
                     n_neighbors=1, niter=200, representation="modes",
                     doPCA=False, n_components=-1,
                     missing=False, missing_strategy='median',
                     random_state=1234, n_folds=5, n_repetitions=10):
    r"""Cross-validate GTR model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_individuals, 1)
        Labels for train set.
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
    s : float, optional (default = -1)
        RBF width factor. Default (-1) is to try different values.
        Parameter to tune width of RBF functions.
        Impacts manifold flexibility.
    regul : float, optional (default = -1)
        Regularization coefficient. Default (-1) is to try different values.
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
    n_folds : int, optional (default = 5)
        Number of CV folds.
    n_repetitions : int, optional (default = 10)
        Number of CV iterations.
    """
    print("")
    print("k = sqrt(grid size), m = sqrt(radial basis function grid size), "
          "regul = regularization, s = RBF width factor")
    print("")
    if k == 0:
        k = int(np.sqrt(5*np.sqrt(data.shape[0])))+2
    if m == 0:
        m = int(np.sqrt(k))
    if n_components == -1 and doPCA is True:
        pca = PCA(random_state=random_state)
        pca.fit(data)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used number of components explaining 80%% of the variance = %s\n"
              % n_components)
    if regul < 0.0:
        lvec = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    else:
        lvec = [regul]
    if s <= 0.0:
        svec = [0.25, 0.5, 1.0, 1.50, 2.0]
    else:
        svec = [s]
    savemean = 999999999
    saveh = 0.0
    modelvec = ""
    savemeanr2 = 0.0
    savehr2 = 0.0
    print("k:m:s:regul\tRMSE with CI\tR2 with CI\t")
    for s in svec:
        for regul in lvec:
            modelstring = str(s)+":"+str(regul)
            rmsevec = []
            r2vec = []
            for j in range(n_repetitions):
                ss = KFold(n_splits=n_folds, shuffle=True, random_state=j)
                y_true = []
                y_pred = []
                for train_index, test_index in ss.split(data):
                    train = np.copy(data[train_index])
                    test = np.copy(data[test_index])
                    prediction = ugtm_predictions.GTR(train=train,
                                                      labels=labels[train_index],
                                                      test=test, k=k,
                                                      m=m, s=s, regul=regul,
                                                      n_neighbors=n_neighbors,
                                                      niter=niter,
                                                      representation=representation,
                                                      doPCA=doPCA,
                                                      n_components=n_components,
                                                      random_state=random_state,
                                                      missing=missing,
                                                      missing_strategy=missing_strategy)
                    y_pred = np.append(y_pred, prediction)
                    y_true = np.append(y_true, labels[test_index])
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                rmsevec = np.append(rmsevec, rmse)
                r2vec = np.append(r2vec, r2)
            mean, se = np.mean(rmsevec), st.sem(rmsevec)
            h = se * t._ppf((1.0+0.95)/2., len(rmsevec)-1)
            meanr2, ser2 = np.mean(r2vec), st.sem(r2vec)
            hr2 = ser2 * t._ppf((1.0+0.95)/2., len(r2vec)-1)
            if(mean < savemean):
                savemean = mean
                saveh = h
                modelvec = modelstring
                savemeanr2, saveser2 = np.mean(r2vec), st.sem(r2vec)
                savehr2 = saveser2 * t._ppf((1+0.95)/2., len(r2vec)-1)
            print("%s\t%.4f +/- %.4f\t%.4f +/- %.4f"
                  % (str(k)+':'+str(m)+':'+modelstring, mean, h, meanr2, hr2))
    print('')
    print("########best GTR model##########")
    print("%s\t%.4f +/- %.4f\t%.4f +/- %.4f"
          % (str(k)+':'+str(m)+':'+modelvec,
             savemean, saveh, savemeanr2, savehr2))
    print("")


def crossvalidatePCAC(data, labels, n_neighbors=1, maxneighbours=11,
                      doPCA=False, n_components=-1, missing=False,
                      missing_strategy='median', random_state=1234,
                      n_folds=5, n_repetitions=10):
    r"""Cross-validate PCA kNN classification model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_individuals, 1)
        Labels for train set.
    n_neighbors : int, optional (default = 1)
        Number of neighbors for kNN algorithm (number of nearest nodes).
    max_neighbors : int, optional (default = 11)
        The function crossvalidates kNN models with k between n_neighbors
        and max_neighbors.
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
    n_folds : int, optional (default = 5)
        Number of CV folds.
    n_repetitions : int, optional (default = 10)
        Number of CV iterations.
    """
    if n_components == -1 and doPCA is True:
        pca = PCA(random_state=random_state)
        pca.fit(data)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used number of components "
              "explaining 80%% of the variance = %s\n"
              % n_components)
    uniqClasses, labels = np.unique(labels, return_inverse=True)
    nClasses = len(uniqClasses)
    print("Classes: ", uniqClasses)
    print("nClasses: ", nClasses)
    print("")
    print("model\tparameters=k_for_kNN\trecall with CI\t"
          "precision with CI\tF1-score with CI")
    print("")
    if n_neighbors <= 0:
        Kvec = np.arange(start=1, stop=maxneighbours, step=1, dtype=np.int32)
    else:
        Kvec = [n_neighbors]

    savemean = -9999
    nummodel = 0
    savemodel = ""
    for c in Kvec:
        nummodel += 1
        modelstring = str(c)
        recallvec = []
        precisionvec = []
        f1vec = []
        recallclassvec = np.array([])
        precisionclassvec = np.array([])
        f1classvec = np.array([])
        meanclass = np.zeros(nClasses)
        meanprecisionclass = np.zeros(nClasses)
        meanf1class = np.zeros(nClasses)
        seclass = np.zeros(nClasses)
        seprecisionclass = np.zeros(nClasses)
        sef1class = np.zeros(nClasses)
        hclass = np.zeros(nClasses)
        hprecisionclass = np.zeros(nClasses)
        hf1class = np.zeros(nClasses)
        for j in range(n_repetitions):
            ss = KFold(n_splits=n_folds, shuffle=True, random_state=j)
            y_true = []
            y_pred = []
            for train_index, test_index in ss.split(data):
                train = np.copy(data[train_index])
                test = np.copy(data[test_index])
                processed = ugtm_preprocess.processTrainTest(train,
                                                             test, doPCA,
                                                             n_components,
                                                             missing,
                                                             missing_strategy)
                y_pred = np.append(y_pred, ugtm_predictions.predictNNSimple(
                    processed.train,
                    processed.test,
                    labels[train_index],
                    c,
                    "classification"))
                y_true = np.append(y_true, labels[test_index])
            recall = recall_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            recallvec = np.append(recallvec, recall)
            precisionvec = np.append(precisionvec, precision)
            f1vec = np.append(f1vec, f1)
            recallclass = recall_score(y_true, y_pred, average=None)
            precisionclass = precision_score(y_true, y_pred, average=None)
            f1class = f1_score(y_true, y_pred, average=None)
            if(j == 0):
                recallclassvec = recallclass
                precisionclassvec = precisionclass
                f1classvec = f1class
            else:
                recallclassvec = np.vstack([recallclassvec, recallclass])
                precisionclassvec = np.vstack(
                    [precisionclassvec, precisionclass])
                f1classvec = np.vstack([f1classvec, f1class])
        mean, se = np.mean(recallvec), st.sem(recallvec)
        meanprecision, seprecision = np.mean(
            precisionvec), st.sem(precisionvec)
        meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
        h = se * t._ppf((1+0.95)/2., len(recallvec)-1)
        hprecision = seprecision * t._ppf((1+0.95)/2., len(precisionvec)-1)
        hf1 = sef1 * t._ppf((1+0.95)/2., len(f1vec)-1)
        if(meanf1 > savemean):
            savemean = meanf1
            savemodel = "Model "+str(nummodel)
        for i in range(0, nClasses):
            meanclass[i] = np.mean(recallclassvec[:, i])
            seclass[i] = st.sem(recallclassvec[:, i])
            meanf1class[i] = np.mean(f1classvec[:, i])
            sef1class[i] = st.sem(f1classvec[:, i])
            meanprecisionclass[i] = np.mean(precisionclassvec[:, i])
            seprecisionclass[i] = st.sem(precisionclassvec[:, i])
            hclass[i] = seclass[i] * \
                t._ppf((1+0.95)/2., len(recallclassvec[:, i])-1)
            hprecisionclass[i] = seprecisionclass[i] \
                * t._ppf((1+0.95)/2.,
                         len(precisionclassvec[:, i])-1)
            hf1class[i] = sef1class[i] * \
                t._ppf((1+0.95)/2., len(f1classvec[:, i])-1)

        print("Model %s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
              % (nummodel, modelstring,
                 mean, h, meanprecision, hprecision, meanf1, hf1))
        for i in range(nClasses):
            print("Class=%s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
                  % (uniqClasses[i], modelstring, meanclass[i], hclass[i],
                      meanprecisionclass[i], hprecisionclass[i],
                      meanf1class[i], hf1class[i]))
        print('')
    print('')
    print("########best nearest neighbors model##########")
    print(savemodel)
    print("")


def crossvalidatePCAR(data, labels, n_neighbors=1,
                      maxneighbours=11, doPCA=False,
                      n_components=-1, missing=False,
                      missing_strategy='median',
                      random_state=1234,
                      n_folds=5, n_repetitions=10):
    r"""Cross-validate PCA kNN regression model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_individuals, 1)
        Labels for train set.
    n_neighbors : int, optional (default = 1)
        Number of neighbors for kNN algorithm (number of nearest nodes).
    max_neighbors : int, optional (default = 11)
        The function crossvalidates kNN models with k between n_neighbors
        and max_neighbors.
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
    n_folds : int, optional (default = 5)
        Number of CV folds.
    n_repetitions : int, optional (default = 10)
        Number of CV iterations.
    """
    if n_components == -1 and doPCA is True:
        pca = PCA(random_state=random_state)
        pca.fit(data)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used number of components explaining 80%% of the variance = %s\n"
              % n_components)
    print("")
    uniqClasses, labels = np.unique(labels, return_inverse=True)
    if n_neighbors <= 0:
        Kvec = np.arange(start=1, stop=maxneighbours, step=1, dtype=np.int32)
    else:
        Kvec = [n_neighbors]

    modelvec = ""
    savemean = 99999
    saveh = 0.0
    savemeanr2 = 0.0
    savehr2 = 0.0
    nummodel = 0
    print("k = number of nearest neighbours\tRMSE with CI\tR2 with CI\t")
    for c in Kvec:
        nummodel += 1
        modelstring = str(c)
        rmsevec = []
        r2vec = []
        for j in range(n_repetitions):
            ss = KFold(n_splits=n_folds, shuffle=True, random_state=j)
            y_true = []
            y_pred = []
            for train_index, test_index in ss.split(data):
                train = np.copy(data[train_index])
                test = np.copy(data[test_index])
                processed = ugtm_preprocess.processTrainTest(train, test,
                                                             doPCA,
                                                             n_components,
                                                             missing,
                                                             missing_strategy)
                y_pred = np.append(y_pred, ugtm_predictions.predictNNSimple(
                    processed.train,
                    processed.test,
                    labels[train_index],
                    c, "regression"))
                y_true = np.append(y_true, labels[test_index])
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            rmsevec = np.append(rmsevec, rmse)
            r2vec = np.append(r2vec, r2)
        mean, se = np.mean(rmsevec), st.sem(rmsevec)
        h = se * t._ppf((1+0.95)/2., len(rmsevec)-1)
        meanr2, ser2 = np.mean(r2vec), st.sem(r2vec)
        hr2 = ser2 * t._ppf((1+0.95)/2., len(r2vec)-1)
        if(mean < savemean):
            savemean = mean
            saveh = h
            modelvec = modelstring
            savemeanr2, saveser2 = np.mean(r2vec), st.sem(r2vec)
            savehr2 = saveser2 * t._ppf((1+0.95)/2., len(r2vec)-1)
        print("%s\t%.4f +/- %.4f\t%.4f +/- %.4f"
              % (modelstring, mean, h, meanr2, hr2))
    print('')
    print("########best nearest neighbors model##########")
    print("%s\t%.4f +/- %.4f\t%.4f +/- %.4f"
          % (modelvec, savemean, saveh, savemeanr2, savehr2))
    print("")


def crossvalidateSVC(data, labels, C=1.0, doPCA=False, n_components=-1,
                     missing=False,
                     missing_strategy='median',
                     random_state=1234, n_folds=5, n_repetitions=10):
    r"""Cross-validate SVC model.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_individuals, 1)
        Labels for train set.
    C : float, optional (default = 1.0)
        SVM regularization parameter.
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
    n_folds : int, optional (default = 5)
        Number of CV folds.
    n_repetitions : int, optional (default = 10)
        Number of CV iterations.
    """

    if n_components == -1 and doPCA is True:
        pca = PCA(random_state=random_state)
        pca.fit(data)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used number of components explaining 80%% of the variance = %s\n"
              % n_components)
    uniqClasses, labels = np.unique(labels, return_inverse=True)
    nClasses = len(uniqClasses)
    print("Classes: ", uniqClasses)
    print("nClasses: ", nClasses)
    print("")
    print("model\tparameters=C\trecall with CI\t"
          "precision with CI\tF1-score with CI")
    print("")
    if C < 0.0:
        Cvec = np.power(2, np.arange(
            start=-5, stop=15, step=1, dtype=np.float))
    else:
        Cvec = [C]
    savemean = -9999
    nummodel = 0
    savemodel = ""
    for C in Cvec:
        modelstring = str(C)
        nummodel += 1
        recallvec = []
        precisionvec = []
        f1vec = []
        recallclassvec = np.array([])
        precisionclassvec = np.array([])
        f1classvec = np.array([])
        meanclass = np.zeros(nClasses)
        meanprecisionclass = np.zeros(nClasses)
        meanf1class = np.zeros(nClasses)
        seclass = np.zeros(nClasses)
        seprecisionclass = np.zeros(nClasses)
        sef1class = np.zeros(nClasses)
        hclass = np.zeros(nClasses)
        hprecisionclass = np.zeros(nClasses)
        hf1class = np.zeros(nClasses)
        for j in range(n_repetitions):
            ss = KFold(n_splits=n_folds, shuffle=True, random_state=j)
            y_true = []
            y_pred = []
            for train_index, test_index in ss.split(data):
                train = np.copy(data[train_index])
                test = np.copy(data[test_index])
                processed = ugtm_preprocess.processTrainTest(train, test,
                                                             doPCA,
                                                             n_components,
                                                             missing,
                                                             missing_strategy)
                clf = SVC(kernel='linear', C=C)
                clf.fit(processed.train, labels[train_index])
                y_pred = np.append(y_pred, clf.predict(processed.test))
                y_true = np.append(y_true, labels[test_index])
            recall = recall_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            recallvec = np.append(recallvec, recall)
            precisionvec = np.append(precisionvec, precision)
            f1vec = np.append(f1vec, f1)
            recallclass = recall_score(y_true, y_pred, average=None)
            precisionclass = precision_score(y_true, y_pred, average=None)
            f1class = f1_score(y_true, y_pred, average=None)
            if(j == 0):
                recallclassvec = recallclass
                precisionclassvec = precisionclass
                f1classvec = f1class
            else:
                recallclassvec = np.vstack([recallclassvec, recallclass])
                precisionclassvec = np.vstack(
                    [precisionclassvec, precisionclass])
                f1classvec = np.vstack([f1classvec, f1class])
        mean, se = np.mean(recallvec), st.sem(recallvec)
        meanprecision, seprecision = np.mean(
            precisionvec), st.sem(precisionvec)
        meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
        h = se * t._ppf((1+0.95)/2., len(recallvec)-1)
        hprecision = seprecision * t._ppf((1+0.95)/2., len(precisionvec)-1)
        hf1 = sef1 * t._ppf((1+0.95)/2., len(f1vec)-1)
        if(meanf1 > savemean):
            savemean = meanf1
            savemodel = "Model "+str(nummodel)
        for i in range(0, nClasses):
            meanclass[i] = np.mean(recallclassvec[:, i])
            seclass[i] = st.sem(recallclassvec[:, i])
            sef1class[i] = st.sem(f1classvec[:, i])
            meanf1class[i] = np.mean(f1classvec[:, i])
            meanprecisionclass[i] = np.mean(precisionclassvec[:, i])
            seprecisionclass[i] = st.sem(precisionclassvec[:, i])
            hclass[i] = seclass[i] * \
                t._ppf((1+0.95)/2., len(recallclassvec[:, i])-1)
            hprecisionclass[i] = seprecisionclass[i] * \
                t._ppf((1+0.95)/2., len(precisionclassvec[:, i])-1)
            hf1class[i] = sef1class[i] * \
                t._ppf((1+0.95)/2., len(f1classvec[:, i])-1)
        print("Model %s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
              % (nummodel, modelstring,
                 mean, h, meanprecision, hprecision, meanf1, hf1))
        for i in range(nClasses):
            print("Class=%s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
                  % (uniqClasses[i], modelstring, meanclass[i], hclass[i],
                     meanprecisionclass[i], hprecisionclass[i],
                     meanf1class[i], hf1class[i]))
        print('')
    print('')
    print("########best linear SVM model##########")
    print(savemodel)
    print("")


def crossvalidateSVR(data, labels,
                     C=-1, epsilon=-1,
                     doPCA=False,
                     n_components=-1, missing=False,
                     missing_strategy='median', random_state=1234,
                     n_folds=5, n_repetitions=10):
    r"""Cross-validate SVR model with linear kernel.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_individuals, 1)
        Labels for train set.
    C : float, optional (default = -1)
        SVM regularization parameter.
        If (C = -1), different values are tested.
    epsilon : float, optional (default = -1)
        SVM tolerance parameter.
        If (epsilon = -1), different values are tested.
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
    n_folds : int, optional (default = 5)
        Number of CV folds.
    n_repetitions : int, optional (default = 10)
        Number of CV iterations.
    """
    if C < 0.0:
        Cvec = np.power(2, np.arange(
            start=-5, stop=15, step=1, dtype=np.float))
    else:
        Cvec = [C]
    if epsilon < 0.0:
        EpsVec = [0, 0.01, 0.1, 0.5, 1, 2, 4]
    else:
        EpsVec = [epsilon]
    modelvec = ""
    savemean = 99999
    saveh = 0.0
    savemeanr2 = 0.0
    savehr2 = 0.0
    if n_components == -1 and doPCA is True:
        pca = PCA(random_state=random_state)
        pca.fit(data)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used number of components explaining 80%%"
              "of the variance = %s\n"
              % n_components)
    print("C:epsilon\tRMSE with CI\tR2 with CI\t")
    for C in Cvec:
        for eps in EpsVec:
            modelstring = str(C)+":"+str(eps)
            rmsevec = []
            r2vec = []
            for j in range(n_repetitions):
                ss = KFold(n_splits=n_folds, shuffle=True, random_state=j)
                y_true = []
                y_pred = []
                for train_index, test_index in ss.split(data):
                    train = np.copy(data[train_index])
                    test = np.copy(data[test_index])
                    processed = ugtm_preprocess.processTrainTest(train, test,
                                                                 doPCA,
                                                                 n_components,
                                                                 missing,
                                                                 missing_strategy)
                    clf = SVR(kernel='linear', C=C, epsilon=eps)
                    clf.fit(processed.train, labels[train_index])
                    y_pred = np.append(y_pred, clf.predict(processed.test))
                    y_true = np.append(y_true, labels[test_index])
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                rmsevec = np.append(rmsevec, rmse)
                r2vec = np.append(r2vec, r2)
            mean, se = np.mean(rmsevec), st.sem(rmsevec)
            h = se * t._ppf((1+0.95)/2., len(rmsevec)-1)
            meanr2, ser2 = np.mean(r2vec), st.sem(r2vec)
            hr2 = ser2 * t._ppf((1+0.95)/2., len(r2vec)-1)
            if(mean < savemean):
                savemean = mean
                saveh = h
                modelvec = modelstring
                savemeanr2, saveser2 = np.mean(r2vec), st.sem(r2vec)
                savehr2 = saveser2 * t._ppf((1+0.95)/2., len(r2vec)-1)
            print("%s\t%.4f +/- %.4f\t%.4f +/- %.4f"
                  % (modelstring, mean, h, meanr2, hr2))
    print('')
    print("########best linear SVM model##########")
    print("%s\t%.4f +/- %.4f\t%.4f +/- %.4f"
          % (modelvec, savemean, saveh, savemeanr2, savehr2))
    print("")


def crossvalidateSVCrbf(data, labels,  C=1, gamma=1, doPCA=False,
                        n_components=-1, missing=False,
                        missing_strategy='median',
                        random_state=1234, n_folds=5,
                        n_repetitions=10):
    r"""Cross-validate SVC model with RBF kernel.

    Parameters
    ==========
    data : array of shape (n_individuals, n_dimensions)
        Train set data matrix.
    labels : array of shape (n_individuals, 1)
        Labels for train set.
    C : float, optional (default = 1)
        SVM regularization parameter.
    gamma : float, optional (default = 1)
        RBF parameter.
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
    n_folds : int, optional (default = 5)
        Number of CV folds.
    n_repetitions : int, optional (default = 10)
        Number of CV iterations.
    """
    if C < 0.0:
        Cvec = np.power(2, np.arange(
            start=-5, stop=15, step=1, dtype=np.float))
    else:
        Cvec = [C]
    if gamma < 0.0:
        gvec = np.power(2.0, np.arange(
            start=-15, stop=3, step=1, dtype=np.float))
    else:
        gvec = [gamma]
#    modelvec = ""
    savemean = -9999.0
#    saveh = 0.0
    nummodel = 0
    if n_components == -1 and doPCA is True:
        pca = PCA(random_state=random_state)
        pca.fit(data)
        n_components = np.searchsorted(
            pca.explained_variance_ratio_.cumsum(), 0.8)+1
        print("Used number of components explaining 80%% "
              "of the variance = %s\n"
              % n_components)
    uniqClasses, labels = np.unique(labels, return_inverse=True)
    nClasses = len(uniqClasses)
    print("Classes: ", uniqClasses)
    print("nClasses: ", nClasses)
    print("")
    print("model\tparameters=C:gamma\trecall with CI\t"
          "precision with CI\tF1-score with CI")
    print("")
    for C in Cvec:
        for g in gvec:
            modelstring = str(C)+"-"+str(g)
            nummodel += 1
            recallvec = []
            precisionvec = []
            f1vec = []
            recallclassvec = np.array([])
            precisionclassvec = np.array([])
            f1classvec = np.array([])
            meanclass = np.zeros(nClasses)
            meanprecisionclass = np.zeros(nClasses)
            meanf1class = np.zeros(nClasses)
            seclass = np.zeros(nClasses)
            seprecisionclass = np.zeros(nClasses)
            sef1class = np.zeros(nClasses)
            hclass = np.zeros(nClasses)
            hprecisionclass = np.zeros(nClasses)
            hf1class = np.zeros(nClasses)
            for j in range(n_repetitions):
                ss = KFold(n_splits=n_folds, shuffle=True, random_state=j)
                y_true = []
                y_pred = []
                for train_index, test_index in ss.split(data):
                    train = np.copy(data[train_index])
                    test = np.copy(data[test_index])
                    processed = ugtm_preprocess.processTrainTest(train, test,
                                                                 doPCA,
                                                                 n_components,
                                                                 missing,
                                                                 missing_strategy)
                    clf = SVC(kernel='rbf', C=C, gamma=g)
                    clf.fit(processed.train, labels[train_index])
                    y_pred = np.append(y_pred, clf.predict(processed.test))
                    y_true = np.append(y_true, labels[test_index])
                recall = recall_score(y_true, y_pred, average='weighted')
                precision = precision_score(
                    y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
                recallvec = np.append(recallvec, recall)
                precisionvec = np.append(precisionvec, precision)
                f1vec = np.append(f1vec, f1)
                recallclass = recall_score(y_true, y_pred, average=None)
                precisionclass = precision_score(y_true, y_pred, average=None)
                f1class = f1_score(y_true, y_pred, average=None)
                if(j == 0):
                    recallclassvec = recallclass
                    precisionclassvec = precisionclass
                    f1classvec = f1class
                else:
                    recallclassvec = np.vstack([recallclassvec, recallclass])
                    precisionclassvec = np.vstack(
                        [precisionclassvec, precisionclass])
                    f1classvec = np.vstack([f1classvec, f1class])
            mean, se = np.mean(recallvec), st.sem(recallvec)
            meanprecision, seprecision = np.mean(
                precisionvec), st.sem(precisionvec)
            meanf1, sef1 = np.mean(f1vec), st.sem(f1vec)
            h = se * t._ppf((1+0.95)/2., len(recallvec)-1)
            hprecision = seprecision * \
                t._ppf((1+0.95)/2., len(precisionvec)-1)
            hf1 = sef1 * t._ppf((1+0.95)/2., len(f1vec)-1)
            if(meanf1 > savemean):
                savemean = meanf1
#                saveh = hf1
#                modelvec = modelstring
                savemodel = "Model "+str(nummodel)
            for i in range(0, nClasses):
                meanclass[i], seclass[i] = np.mean(recallclassvec[:, i]), \
                    st.sem(recallclassvec[:, i])
                meanf1class[i], sef1class[i] = np.mean(f1classvec[:, i]), \
                    st.sem(f1classvec[:, i])
                meanprecisionclass[i] = np.mean(precisionclassvec[:, i])
                seprecisionclass[i] = st.sem(precisionclassvec[:, i])
                hclass[i] = seclass[i] * \
                    t._ppf((1+0.95)/2., len(recallclassvec[:, i])-1)
                hprecisionclass[i] = seprecisionclass[i] * \
                    t._ppf((1+0.95)/2., len(precisionclassvec[:, i])-1)
                hf1class[i] = sef1class[i] * \
                    t._ppf((1+0.95)/2., len(f1classvec[:, i])-1)
            print("Model %s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
                  % (nummodel, modelstring, mean, h,
                     meanprecision, hprecision, meanf1, hf1))
            for i in range(nClasses):
                print("Class=%s\t%s\t%.4f +/- %.4f\t%.4f +/- %.4f\t%.4f +/- %.4f"
                      % (uniqClasses[i], modelstring, meanclass[i],
                         hclass[i], meanprecisionclass[i],
                         hprecisionclass[i], meanf1class[i],
                         hf1class[i]))
            print("")
    print("")

    print("########best RBF SVM model##########")
    print(savemodel)
    print("")


def whichExperiment(data, labels, args, discrete=False):
    if discrete is True and args.model == 'GTM':
        decide = 'crossvalidateGTC'
    elif discrete is False and args.model == 'GTM':
        decide = 'crossvalidateGTR'
    elif discrete is True and args.model == 'SVM':
        decide = 'crossvalidateSVC'
    elif discrete is False and args.model == 'SVM':
        decide = 'crossvalidateSVR'
    elif discrete is True and args.model == 'SVMrbf':
        decide = 'crossvalidateSVCrbf'
    elif discrete is True and args.model == 'PCA':
        decide = 'crossvalidatePCAC'
    elif discrete is False and args.model == 'PCA':
        decide = 'crossvalidatePCAR'
    elif discrete is True and args.model == 'compare':
        decide = 'comparecrossvalidateC'
    elif discrete is False and args.model == 'compare':
        decide = 'comparecrossvalidateR'
    else:
        decide = ''
        exit
    if decide == 'crossvalidateGTC':
        crossvalidateGTC(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         n_neighbors=args.n_neighbors, representation=args.representation,
                         missing=args.missing, missing_strategy=args.missing_strategy,
                         random_state=args.random_state, k=args.grid_size,
                         m=args.rbf_grid_size, predict_mode=args.predict_mode,
                         prior=args.prior, regul=args.regularization,
                         s=args.rbf_width_factor)
    elif decide == 'crossvalidateGTR':
        crossvalidateGTR(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         n_neighbors=args.n_neighbors, representation=args.representation,
                         missing=args.missing, missing_strategy=args.missing_strategy,
                         random_state=args.random_state, k=args.grid_size, m=args.rbf_grid_size,
                         regul=args.regularization, s=args.rbf_width_factor)
    elif decide == 'crossvalidateSVC':
        crossvalidateSVC(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         missing=args.missing, missing_strategy=args.missing_strategy,
                         random_state=args.random_state, C=args.svm_margin)
    elif decide == 'crossvalidateSVCrbf':
        crossvalidateSVCrbf(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                            missing=args.missing, missing_strategy=args.missing_strategy,
                            random_state=args.random_state, C=args.svm_margin, gamma=args.svm_gamma)
    elif decide == 'crossvalidateSVR':
        crossvalidateSVR(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         missing=args.missing, missing_strategy=args.missing_strategy,
                         random_state=args.random_state, C=args.svm_margin, epsilon=args.svm_epsilon)
    elif decide == 'crossvalidatePCAC':
        crossvalidatePCAC(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                          missing=args.missing, missing_strategy=args.missing_strategy,
                          random_state=args.random_state, n_neighbors=args.n_neighbors)
    elif decide == 'crossvalidatePCAR':
        crossvalidatePCAR(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                          missing=args.missing, missing_strategy=args.missing_strategy,
                          random_state=args.random_state, n_neighbors=args.n_neighbors)
    elif decide == 'comparecrossvalidateC':
        crossvalidateSVC(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         missing=args.missing, missing_strategy=args.missing_strategy,
                         random_state=args.random_state, C=args.svm_margin)
        crossvalidateGTC(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         representation=args.representation, missing=args.missing,
                         missing_strategy=args.missing_strategy, random_state=args.random_state,
                         k=args.grid_size, m=args.rbf_grid_size, predict_mode=args.predict_mode,
                         prior=args.prior, regul=args.regularization, s=args.rbf_width_factor)
    elif decide == 'comparecrossvalidateR':
        crossvalidateSVR(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         missing=args.missing, missing_strategy=args.missing_strategy,
                         random_state=args.random_state,
                         C=args.svm_margin, epsilon=args.svm_epsilon)
        crossvalidateGTR(data=data, labels=labels, doPCA=args.pca, n_components=args.n_components,
                         n_neighbors=args.n_neighbors, representation=args.representation,
                         missing=args.missing, missing_strategy=args.missing_strategy,
                         random_state=args.random_state, k=args.grid_size, m=args.rbf_grid_size,
                         regul=args.regularization, s=args.rbf_width_factor)
    else:
        print("Could not determine which experiment to conduct.")
