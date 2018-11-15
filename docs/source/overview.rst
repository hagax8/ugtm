========
Overview
========


Generative topographic mapping (GTM) is a probabilisitc dimensionality reduction algorithm introduced by `Bishop, Svensen and Williams <https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf>`_, which can also be used for classification and regression using class maps or activity landscapes: 

.. altair-plot::
        :hide-code:  

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

        # standardize 
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

        # Construct class map 
        gtc = eGTC(prior="equiprobable") 
        gtc = gtc.fit(X,y)

        points = eGTM().fit(X).transform(X)

        df = pd.DataFrame(points, columns=["x1", "x2"])
        df['original label'] = iris.target_names[y]

        dfclassmap = pd.DataFrame(gtc.optimizedModel.matX, columns=["x1", "x2"]) 
        dfclassmap["predicted node label"] = iris.target_names[gtc.node_label]
        dfclassmap["probability of predominant class"] = np.max(gtc.node_probabilities,axis=1) 

        # Classification map
        ch1 = alt.Chart(dfclassmap).mark_square().encode(
            x='x1',
            y='x2',
            color=alt.Color('predicted node label:N',legend=alt.Legend(title="label")),
            size=alt.value(80),
            opacity='probability of predominant class',
            tooltip=['x1','x2', 'predicted node label:N', 'probability of predominant class']
        ).properties(title = "GTM class map: iris dataset",
                     width = 200, height = 200)

        ch2 = alt.Chart(df).mark_circle(stroke="grey").encode(
           x='x1',
           y='x2',
           color=alt.Color('original label:N',legend=alt.Legend(title="label")),
           tooltip=['original label','x1','x2']
        )

        chr3 = ch1 + ch2
        chr3.configure_axis(grid=False).configure_view(strokeOpacity=0)

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


