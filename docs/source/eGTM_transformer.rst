======================
eGTM: GTM transformer
======================

Run GTM
-------

:class:`~ugtm.ugtm_sklearn.eGTM` is a sklearn-compatible GTM transformer. Similarly to PCA or t-SNE, eGTM reduces the dimensionality from n_dimensions to 2 dimensions. To generate mean GTM 2D projections::

        from ugtm import eGTM
        import numpy as np

        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(50, 50)

        # Fit GTM on X_train and get 2D projections for X_test
        transformed = eGTM().fit(X_train).transform(X_test)

The default output of eGTM.transform is the mean GTM projection. For other data representations (modes, responsibilities), see :func:`~ugtm.ugtm_sklearn.eGTM.transform`.



Visualize projection
--------------------

Visualization demo using altair https://altair-viz.github.io:

.. altair-plot::

        from ugtm import eGTM
        import numpy as np
        import altair as alt
        import pandas as pd

        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(50, 50)

        transformed = eGTM().fit(X_train).transform(X_test)

        df = pd.DataFrame(transformed, columns=["x1", "x2"])
        alt.Chart(df).mark_point().encode(
        x='x1',y='x2',
        tooltip=["x1", "x2"]
        ).properties(title="GTM projection of X_test").interactive()






