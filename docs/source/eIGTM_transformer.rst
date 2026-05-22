========================
eIGTM: incremental GTM
========================

Overview
--------

:class:`~ugtm.ugtm_sklearn.eIGTM` is a memory-efficient, sklearn-compatible
GTM transformer for large datasets (Gaspar et al. 2014).
Standard GTM holds the full N×K responsibility matrix in RAM at every EM
iteration. iGTM processes the data in blocks, accumulating only two small
sufficient-statistic arrays per iteration:

* **g_vec** — shape (n_nodes,): accumulated row sums of R across blocks
* **RT_acc** — shape (n_nodes, n_dimensions): accumulated R @ X across blocks

Peak memory per iteration is therefore O(block_size × n_nodes) rather than
O(N × n_nodes). The W update is mathematically equivalent to standard GTM;
only the β⁻¹ update uses the block-local distances rather than recomputed
distances, which introduces a minor approximation that has negligible effect
on the final manifold (Pearson r > 0.999 between GTM and iGTM coordinates
in practice).


Run eIGTM
----------

The API mirrors :class:`~ugtm.ugtm_sklearn.eGTM` with one extra parameter
``n_blocks`` (0 = auto, set to ``ceil(N / 5000)``)::

        from ugtm import eIGTM
        import numpy as np

        X_train = np.random.randn(10000, 50)
        X_test  = np.random.randn(1000, 50)

        # Fit iGTM on X_train; blocks chosen automatically
        model = eIGTM().fit(X_train)

        # 2D projection of X_test
        transformed = model.transform(X_test)

        # Or fit and transform in one call
        transformed_train = eIGTM().fit_transform(X_train)


For the low-level wrapper (mirrors :func:`~ugtm.ugtm_gtm.runGTM`)::

        from ugtm import runIGTM
        import numpy as np

        data = np.random.randn(10000, 50)
        model = runIGTM(data, n_blocks=5)

        # Access 2D coordinates
        coordinates = model.matMeans
        modes       = model.matModes


Block-wise projection for large test sets
------------------------------------------

When the test set is also large, use the
:meth:`~ugtm.ugtm_sklearn.eIGTM.transform_blocks` generator so that peak
memory stays bounded::

        from ugtm import eIGTM
        import numpy as np

        X_train = np.random.randn(10000, 50)
        X_test  = np.random.randn(10000, 50)

        model = eIGTM().fit(X_train)

        # Yields one (block_size, 2) array at a time
        for block_coords in model.transform_blocks(X_test, block_size=1000):
            # process or write block_coords here
            pass

        # To collect all at once (only if result fits in RAM):
        import numpy as np
        coords = np.vstack(list(model.transform_blocks(X_test, block_size=1000)))

For ``model='responsibilities'``, ``transform_blocks`` yields
(block_size, n_nodes) arrays, avoiding the N×K matrix entirely::

        model = eIGTM(model='responsibilities').fit(X_train)
        for resp_block in model.transform_blocks(X_test, block_size=1000):
            # resp_block.shape == (1000, n_nodes)
            pass


Choosing n_blocks
-----------------

With the default ``n_blocks=0``, blocks are sized to ~5 000 rows each.
For a dataset of N rows:

.. list-table::
   :header-rows: 1
   :widths: 20 20

   * - N
     - Auto n_blocks
   * - ≤ 5 000
     - 1  (identical to standard GTM)
   * - 10 000
     - 2
   * - 50 000
     - 10
   * - 1 000 000
     - 200

Setting ``n_blocks=1`` reproduces standard GTM behaviour (with the minor
β⁻¹ approximation noted above).


Visualize projection
--------------------

.. altair-plot::

        from ugtm import eIGTM
        import numpy as np
        import altair as alt
        import pandas as pd

        np.random.seed(0)
        X_train = np.random.randn(100, 10)
        X_test  = np.random.randn(50, 10)
        labels  = np.random.choice(['A', 'B', 'C'], size=50)

        transformed = eIGTM(n_blocks=2).fit(X_train).transform(X_test)

        df = pd.DataFrame(transformed, columns=["x1", "x2"])
        df["label"] = labels

        alt.Chart(df).mark_point(size=60).encode(
            x='x1', y='x2',
            color=alt.Color('label:N', scale=alt.Scale(scheme='set1')),
            tooltip=['x1', 'x2', 'label']
        ).properties(title="iGTM projection of X_test", width=300, height=300).interactive()


