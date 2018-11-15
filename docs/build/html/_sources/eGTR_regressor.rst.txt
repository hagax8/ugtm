===================
eGTR: GTM regressor 
===================

Run eGTR
--------

:class:`~ugtm.ugtm_sklearn.eGTR` is a sklearn-compatible GTM regressor. Similarly to PCA or t-SNE, GTM reduces the dimensionality from n_dimensions to 2 dimensions. GTR uses a GTM class map to predict labels for new data (cf. :func:`~gtm.ugtm_landscape.landscape`). The following example uses the iris dataset::

        from ugtm import eGTR
        from sklearn import datasets
        from sklearn import preprocessing
        from sklearn import decomposition 
        from sklearn import model_selection

        boston = datasets.load_boston()
        X = boston.data 
        y = boston.target

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42)

        # optional preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Predict labels for X_test
        gtr = eGTR() 
        gtr = gtr.fit(X_train,y_train)
        y_pred = gtr.predict(X_test)


Visualize activity landscape 
----------------------------

The GTR algorithm is based on an activity landscape. This landscape is discretized into a grid of nodes,
which can be colored by predicted label. This visualization uses the python package `altair <https://altair-viz.github.io>`_:

.. altair-plot::

        from ugtm import eGTR, eGTM
        import numpy as np
        import altair as alt
        import pandas as pd
        from sklearn import datasets
        from sklearn import preprocessing
        from sklearn import decomposition
        from sklearn import metrics
        from sklearn import model_selection

        boston = datasets.load_boston()
        X = boston.data 
        y = boston.target

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42)

        # optional preprocessing
        std = preprocessing.StandardScaler()
        X_train = std.fit(X_train).transform(X_train) 

        # Construct activity landscape 
        gtr = eGTR() 
        gtr = gtr.fit(X_train,y_train)

        dfclassmap = pd.DataFrame(gtr.optimizedModel.matX, columns=["x1", "x2"]) 
        dfclassmap["predicted node label"] = gtr.node_label

        # Classification map
        alt.Chart(dfclassmap).mark_square().encode(
            x='x1',
            y='x2',
            color=alt.Color('predicted node label:Q',
                            scale=alt.Scale(scheme='greenblue'),
                            legend=alt.Legend(title="Boston house prices")),
            size=alt.value(50),
            tooltip=['x1','x2', 'predicted node label:Q']
        ).properties(title = "Activity landscape", width = 200, height = 200)


Visualize predicted vs real labels
----------------------------------

This visualization uses the python package `altair <https://altair-viz.github.io>`_:

.. altair-plot::

        from ugtm import eGTM, eGTR
        import numpy as np
        import altair as alt
        import pandas as pd
        from sklearn import datasets
        from sklearn import preprocessing
        from sklearn import decomposition
        from sklearn import metrics
        from sklearn import model_selection

        boston = datasets.load_boston()
        X = boston.data 
        y = boston.target

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42)

        # optional preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Predict labels for X_test
        gtr = eGTR() 
        gtr = gtr.fit(X_train,y_train)
        y_pred = gtr.predict(X_test)

        # Get GTM transform for X_test
        transformed = eGTM().fit(X_train).transform(X_test)

        df = pd.DataFrame(transformed, columns=["x1", "x2"])
        df["predicted label"] = y_pred
        df["true label"] = y_test

        chart1 = alt.Chart(df).mark_point().encode(
        x='x1',y='x2',
        color=alt.Color("predicted label:Q",scale=alt.Scale(scheme='greenblue'),
                         legend=alt.Legend(title="Boston house prices")),  
        tooltip=["x1", "x2", "predicted label:Q", "true label:Q"]
        ).properties(title="Pedicted labels", width=200, height=200).interactive()

        chart2 = alt.Chart(df).mark_point().encode(
        x='x1',y='x2',
        color=alt.Color("true label:Q",scale=alt.Scale(scheme='greenblue'),
                        legend=alt.Legend(title="Boston house prices")),   
        tooltip=["x1", "x2", "predicted label:Q", "true label:Q"]
        ).properties(title="True labels", width=200, height=200).interactive()

        alt.hconcat(chart1, chart2)



Parameter optimization
----------------------

GridSearchCV from sklearn can be used with eGTC for parameter optimization::

        from ugtm import eGTR
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

        # GTM classifier (GTR) 
        gs = GridSearchCV(eGTR(), tuned_params, cv=3, iid=False, scoring='r2')
        gs.fit(X_train, y_train)
        print(gs.best_params_)
