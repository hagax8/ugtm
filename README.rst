tutorial
========

sklearn integration
-------------------

ugtm v2.0 provides sklearn-compatible GTM transformer (eGTM), GTM classifier (eGTC) and GTM regressor (eGTR)::


        from ugtm import eGTM, eGTC, eGTR
        import numpy as np

        # Dummy train and test
        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(50, 50)
        y_train = np.random.choice([1, 2, 3], size=100)

        # GTM transformer
        transformed = eGTM().fit(X_train).transform(X_test)

        # Predict new labels using GTM classifier (GTC)
        predicted_labels = eGTC().fit(X_train, y_train).predict(X_test)

        # Predict new continuous outcomes using GTM regressor (GTR) 
        predicted_labels = eGTR().fit(X_train, y_train).predict(X_test)


The following sections will show functions no defined within the sklearn framework.


Basic functions
---------------

ugtm provides an implementation of GTM (Generative Topographic Mapping), kGTM (kernel Generative Topographic Mapping), GTM classification models (kNN, Bayes) and GTM regression models. ugtm also implements cross-validation options which can be used to compare GTM classification models to SVM classification models, and GTM regression models to SVM regression models. Typical usage::

    #!/usr/bin/env python

    import ugtm 
    import numpy as np
    
    #generate sample data and labels: replace this with your own data
    data=np.random.randn(100,50)
    labels=np.random.choice([1,2],size=100)

    #build GTM map
    gtm=ugtm.runGTM(data=data,verbose=True)

    #plot GTM map (html)
    gtm.plot_html(output="out")

For installation instructions, cf. https://github.com/hagax8/ugtm


Construct and plot GTM maps (or kGTM maps)
------------------------------------------

A gtm object can be created by running the function runGTM on a dataset. Parameters for runGTM are: k = sqrt(number of nodes), m = sqrt(number of rbf centres), s = RBF width factor, regul = regularization coefficient. The number of iteration for the expectation-maximization algorithm is set to 200 by default. This is an example with random data::

    import ugtm
    
    #import numpy to generate random data
    import numpy as np

    #generate random data (independent variables x), 
    #discrete labels (dependent variable y),
    #and continuous labels (dependent variable y), 
    #to experiment with categorical or continuous outcomes
    
    train = np.random.randn(20,10)
    test = np.random.randn(20,10)
    labels=np.random.choice(["class1","class2"],size=20)
    activity=np.random.randn(20,1)

    #create a gtm object and write model
    gtm = ugtm.runGTM(train)
    gtm.write("testout1")

    #run verbose
    gtm = ugtm.runGTM(train, verbose=True)

    #to run a kernel GTM model instead, run following:
    gtm = ugtm.runkGTM(train, doKernel=True, kernel="linear")

    #access coordinates (means or modes), and responsibilities of gtm object
    gtm_coordinates = gtm.matMeans
    gtm_modes = gtm.matModes
    gtm_responsibilities = gtm.matR



Plot html maps
--------------

Call the plot_html() function on the gtm object::

    #run model on train
    gtm = ugtm.runGTM(train)

    # ex. plot gtm object with landscape, html: labels are continuous
    gtm.plot_html(output="testout10",labels=activity,discrete=False,pointsize=20)

    # ex. plot gtm object with landscape, html: labels are discrete
    gtm.plot_html(output="testout11",labels=labels,discrete=True,pointsize=20)

    # ex. plot gtm object with landscape, html: labels are continuous
    # no interpolation between nodes
    gtm.plot_html(output="testout12",labels=activity,discrete=False,pointsize=20, \
                  do_interpolate=False,ids=labels)

    # ex. plot gtm object with landscape, html: labels are discrete, 
    # no interpolation between nodes
    gtm.plot_html(output="testout13",labels=labels,discrete=True,pointsize=20, \
                  do_interpolate=False)


Plot pdf maps
-------------

Call the plot() function on the gtm object::

    #run model on train
    gtm = ugtm.runGTM(train)

    # ex. plot gtm object, pdf: no labels
    gtm.plot(output="testout6",pointsize=20)

    # ex. plot gtm object with landscape, pdf: labels are discrete
    gtm.plot(output="testout7",labels=labels,discrete=True,pointsize=20)

    # ex. plot gtm object with landscape, pdf: labels are continuous
    gtm.plot(output="testout8",labels=activity,discrete=False,pointsize=20)



Plot multipanel views
---------------------

Call the plot_multipanel() function on the gtm object.
This plots a general model view, showing means, modes, landscape with or without points.
The plot_multipanel function only works if you have defined labels::

    #run model on train
    gtm = ugtm.runGTM(train)

    # ex. with discrete labels and inter-node interpolation
    gtm.plot_multipanel(output="testout2",labels=labels,discrete=True,pointsize=20)

    # ex. with continuous labels and inter-node interpolation
    gtm.plot_multipanel(output="testout3",labels=activity,discrete=False,pointsize=20)

    # ex. with discrete labels and no inter-node interpolation
    gtm.plot_multipanel(output="testout4",labels=labels,discrete=True,pointsize=20, \
                        do_interpolate=False)

    # ex. with continuous labels and no inter-node interpolation
    gtm.plot_multipanel(output="testout5",labels=activity,discrete=False,pointsize=20, \
                        do_interpolate=False)


Project new data onto existing GTM map
--------------------------------------

New data can be projected on the GTM map by using the transform() function, which takes as input the gtm model, a training and test set. The train set is then only used to perform data preprocessing on the test set based on the train (for example: apply the same PCA transformation to the train and test sets before running the algorithm)::

    #run model on train
    gtm = ugtm.runGTM(train,doPCA=True)

    #test new data (test)
    transformed=ugtm.transform(optimizedModel=gtm,train=train,test=test,doPCA=True)

    #plot transformed test (html)
    transformed.plot_html(output="testout14",pointsize=20)

    #plot transformed test (pdf)
    transformed.plot(output="testout15",pointsize=20)

    #plot transformed data on existing classification model, 
    #using training set labels
    gtm.plot_html_projection(output="testout16",projections=transformed,\
                             labels=labels, \
                             discrete=True,pointsize=20)


7. Output predictions for a test set: GTM regression (GTR) and classification (GTC)
====================================================================================

The GTR() function implements the GTM regression model (cf. references) and GTC() function a GTM classification model (cf. references)::

    #continuous labels (prediction by GTM regression model)
    predicted=ugtm.GTR(train=train,test=test,labels=activity)

    #discrete labels (prediction by GTM classification model)
    predicted=ugtm.GTC(train=train,test=test,labels=labels)


8. Advanced GTM predictions with per-class probabilities
=========================================================

Per-class probabilities for a test set can be given by the advancedGTC() function (you can set the m, k, regul, s parameters just as with runGTM)::

    #get whole output model and label predictions for test set
    predicted_model=ugtm.advancedGTC(train=train,test=test,labels=labels)

    #write whole predicted model with per-class probabilities
    ugtm.printClassPredictions(predicted_model,"testout17")



9. Crossvalidation experiments
==============================

Different crossvalidation experiments were implemented to compare GTC and GTR models to classical machine learning methods::

    #crossvalidation experiment: GTM classification model implemented in ugtm, 
    #here: set hyperparameters s=1 and regul=1 (set to -1 to optimize)
    ugtm.crossvalidateGTC(data=train,labels=labels,s=1,regul=1,n_repetitions=10,n_folds=5)

    #crossvalidation experiment: GTM regression model
    ugtm.crossvalidateGTR(data=train,labels=activity,s=1,regul=1)

    #you can also run the following functions to compare
    #with other classification/regression algorithms:

    #crossvalidation experiment, k-nearest neighbours classification
    #on 2D PCA map with 7 neighbors (set to -1 to optimize number of neighbours)
    ugtm.crossvalidatePCAC(data=train,labels=labels,n_neighbors=7)

    #crossvalidation experiment, SVC rbf classification model (sklearn implementation):
    ugtm.crossvalidateSVCrbf(data=train,labels=labels,C=1,gamma=1)

    #crossvalidation experiment, linear SVC classification model (sklearn implementation):
    ugtm.crossvalidateSVC(data=train,labels=labels,C=1)

    #crossvalidation experiment, linear SVC regression model (sklearn implementation):
    ugtm.crossvalidateSVR(data=train,labels=activity,C=1,epsilon=1)

    #crossvalidation experiment, k-nearest neighbours regression on 2D PCA map with 7 neighbors:
    ugtm.crossvalidatePCAR(data=train,labels=activity,n_neighbors=7)



10. Links & references
=======================

1. GTM algorithm by Bishop et al: https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf

2. kernel GTM: https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2010-44.pdf

3. GTM classification models: https://www.ncbi.nlm.nih.gov/pubmed/24320683

4. GTM regression models: https://www.ncbi.nlm.nih.gov/pubmed/27490381

5. github: https://github.com/hagax8/ugtm
