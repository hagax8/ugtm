====================
eGTC: GTM classifier 
====================

Run eGTC 
---------
:class:`~ugtm.ugtm_sklearn.eGTC` is a sklearn-compatible GTM classifier. Similarly to PCA or t-SNE, GTM reduces the dimensionality from n_dimensions to 2 dimensions. GTC uses a GTM class map to predict labels for new data (cf. :func:`~gtm.ugtm_landscape.classMap`).
Two algorithms are available: the bayesian classifier GTC (:class:`~gtm.ugtm_sklearn.uGTC`) or the nearest node classifier (:class:`~gtm.ugtm_sklearn.uGTCnn`). The following example uses the iris dataset::

        from ugtm import eGTC
        from sklearn import datasets
        from sklearn import preprocessing
        from sklearn import decomposition 
        from sklearn import metrics
        from sklearn import model_selection

        iris = datasets.load_iris()
        X = iris.data 
        y = iris.target

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42)

        # optional preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Predict labels for X_test
        gtc = eGTC() 
        gtc = gtc.fit(X_train,y_train)
        y_pred = gtc.predict(X_test)

        # Print score
        print(metrics.matthews_corrcoef(y_test,y_pred))


Visualize class map
-------------------

The GTC algorithm is based on a classification map, discretized into a grid of nodes,
which are colored by predicted label. To each node is associated class probabilities:

.. altair-plot::

        from ugtm import eGTM, eGTC
        import numpy as np
        import altair as alt
        import pandas as pd
        from sklearn import datasets
        from sklearn import preprocessing
        from sklearn import decomposition
        from sklearn import metrics
        from sklearn import model_selection

        iris = datasets.load_iris()
        X = iris.data 
        y = iris.target

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42)

        # optional preprocessing
        std = preprocessing.StandardScaler()
        X_train = std.fit(X_train).transform(X_train) 

        # Construct class map 
        gtc = eGTC() 
        gtc = gtc.fit(X_train,y_train)

        dfclassmap = pd.DataFrame(gtc.optimizedModel.matX, columns=["x1", "x2"]) 
        dfclassmap["predicted node label"] = iris.target_names[gtc.node_label]
        dfclassmap["probability of predominant class"] = np.max(gtc.node_probabilities,axis=1) 

        # Classification map
        alt.Chart(dfclassmap).mark_square().encode(
            x='x1',
            y='x2',
            color='predicted node label:N',
            size=alt.value(50),
            opacity='probability of predominant class',
            tooltip=['x1','x2', 'predicted node label:N', 'probability of predominant class']
        ).properties(title = "Class map", width = 200, height = 200)



Visualize predicted vs real labels
----------------------------------

Visualize predicted vs real labels using the iris dataset and `altair <https://altair-viz.github.io>`_:

.. altair-plot::

        from ugtm import eGTM, eGTC
        import numpy as np
        import altair as alt
        import pandas as pd
        from sklearn import datasets
        from sklearn import preprocessing
        from sklearn import decomposition
        from sklearn import model_selection
        from sklearn.metrics import confusion_matrix

        iris = datasets.load_iris()
        X = iris.data 
        y = iris.target

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42)

        # optional preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Predict labels for X_test
        gtc = eGTC() 
        gtc = gtc.fit(X_train,y_train)
        y_pred = gtc.predict(X_test)

        # Get GTM transform for X_test
        transformed = eGTM().fit(X_train).transform(X_test)

        df = pd.DataFrame(transformed, columns=["x1", "x2"])
        df["predicted label"] = iris.target_names[y_pred]
        df["true label"] = iris.target_names[y_test]
        df["probability of predominant class"] = np.max(gtc.posteriors,axis=1) 

        # Projection of X_test colored by predicted label
        chart1 = alt.Chart().mark_circle().encode(
            x='x1',y='x2',
            size=alt.value(100),
            color=alt.Color("predicted label:N",
                   legend=alt.Legend(title="label")), 
            opacity="probability of predominant class:Q", 
            tooltip=["x1", "x2", "predicted label:N",
                     "true label:N", "probability of predominant class"]
        ).properties(title="Pedicted labels", width=200, height=200).interactive()

        # Projection of X_test colored by true label
        chart2 = alt.Chart().mark_circle().encode(
            x='x1', y='x2',
            color=alt.Color("true label:N",
                            legend=alt.Legend(title="label")),
            size=alt.value(100), 
            tooltip=["x1", "x2", "predicted label:N",
                     "true label:N", "probability of predominant class"]
        ).properties(title="True labels", width=200, height=200).interactive()

                
        alt.hconcat(chart1, chart2, data=df)


Parameter optimization
----------------------

GridSearchCV can be used with eGTC for parameter optimization::

        from ugtm import eGTC
        import numpy as np
        from sklearn.model_selection import GridSearchCV

        # Dummy train and test
        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(50, 50)
        y_train = np.random.choice([1, 2, 3], size=100)

        # Parameters to tune
        tuned_params = {'regul': [0.0001, 0.001, 0.01],
                        's': [0.1, 0.2, 0.3],
                        'k': [16],
                        'm': [4]}

        # GTM classifier (GTC), bayesian 
        gs = GridSearchCV(eGTC(), tuned_params, cv=3, iid=False, scoring='accuracy')
        gs.fit(X_train, y_train)
        print(gs.best_params_)
