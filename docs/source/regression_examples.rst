Regression examples
===================

Wine quality
------------

We use the wine quality dataset from http://archive.ics.uci.edu/.

Example of parameter selection and cross-validation using GTM regression (GTR) and SVM classification (SVR)::

        from ugtm import eGTR
        import numpy as np
        from numpy import sqrt
        from sklearn import model_selection
        from sklearn.metrics import mean_squared_error 
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.dummy import DummyRegressor
        import pandas as pd

        # Load red wine data
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        data = pd.read_csv(url,sep=";")
        y = data['quality']
        X = data.drop(labels='quality',axis=1)


        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True, random_state=42)


        std = StandardScaler().fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)

        performances = {}

        # GTM classifier (GTR), bayesian 

        tuned_params = {'regul': [0.0001, 0.001, 0.01, 0.1, 1],
                's': [0.1, 0.2, 0.3],
                'k': [25],
                'm': [5]}

        gs = model_selection.GridSearchCV(eGTR(), tuned_params, cv=3, iid=False, scoring='neg_mean_squared_error')

        gs.fit(X_train, y_train)

        # Returns best score and best parameters
        print(gs.best_score_) 
        print(gs.best_params_) 

        # Test data using model built with best parameters
        y_true, y_pred = y_test, gs.predict(X_test)

        # Record performance on test set (RMSE) 
        performances['gtr'] = sqrt(mean_squared_error(y_true, y_pred))

        # SVM regressor (SVR)

        tuned_params = {'C':[1,10,100,1000],
                'gamma':[1,0.1,0.001,0.0001],
                'kernel':['rbf']}

        gs = model_selection.GridSearchCV(SVR(), tuned_params, cv=3, iid=False, scoring='neg_mean_squared_error')

        gs.fit(X_train, y_train)

        # Returns best score and best parameters
        print(gs.best_score_) 
        print(gs.best_params_)

        # Test data using model built with best parameters
        y_true, y_pred = y_test, gs.predict(X_test)

        # Record performance on test set 
        performances['svm'] = sqrt(mean_squared_error(y_test, y_pred))

        # Create a dummy regressor
        dummy = DummyRegressor(strategy='mean')

        # Train dummy regressor
        dummy.fit(X_train, y_train)
        y_true, y_pred = y_test, dummy.predict(X_test)

        # Dummy performance
        performances['dummy'] = sqrt(mean_squared_error(y_test, y_pred)) 


