Classification examples
=======================


Breast cancer
-------------


We use the breast cancer wisconsin dataset loaded from sklearn, downloaded from https://goo.gl/U2Uwz2.

The variables are the following:

        a. radius (mean of distances from center to points on the perimeter) 
        b. texture (standard deviation of gray-scale values) 
        c. perimeter 
        d. area 
        e. smoothness (local variation in radius lengths) 
        f. compactness (perimeter^2 / area - 1.0) 
        g. concavity (severity of concave portions of the contour) 
        h. concave points (number of concave portions of the contour) 
        i. symmetry 
        j. fractal dimension ("coastline approximation" - 1)

The target variable is the diagnosis (malignant/benign).

Example of parameter selection and cross-validation using GTM classification (GTC) and SVM classification (SVC)::

        from ugtm import eGTC
        from sklearn.datasets import load_breast_cancer  
        import numpy as np
        from sklearn import model_selection
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report


        data = load_breast_cancer()
        X = data.data
        y = data.target

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=True)

        performances = {}


        # GTM classifier (GTC), bayesian 

        tuned_params = {'regul': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                        's': [0.1, 0.2, 0.3],
                        'k': [16],
                        'm': [4]}

        gs = model_selection.GridSearchCV(eGTC(), tuned_params, cv=3, iid=False, scoring='balanced_accuracy')

        gs.fit(X_train, y_train)

        # Returns best score and best parameters
        print(gs.best_score_) 
        print(gs.best_params_) 

        # Test data using model built with best parameters
        y_true, y_pred = y_test, gs.predict(X_test)
        print(classification_report(y_true, y_pred))

        # Record performance on test set 
        performances['gtc'] = balanced_accuracy_score(y_true, y_pred) 


        # SVM classifier (SVC)

        tuned_params = {'C':[1,10,100,1000],
                        'gamma':[1,0.1,0.001,0.0001],
                        'kernel':['rbf']}

        gs = model_selection.GridSearchCV(SVC(random_state=42), tuned_params, cv=3, iid=False, scoring='balanced_accuracy')

        gs.fit(X_train, y_train)

        # Returns best score and best parameters
        print(gs.best_score_) 
        print(gs.best_params_)

        # Test data using model built with best parameters
        y_true, y_pred = y_test, gs.predict(X_test)
        print(classification_report(y_true, y_pred))

        # Record performance on test set
        performances['svm'] = balanced_accuracy_score(y_test, y_pred) 

        # Algorithm with best performance 
        max(performances.items(), key = lambda x: x[1])
