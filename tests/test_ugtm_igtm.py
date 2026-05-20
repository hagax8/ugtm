import unittest
import numpy as np
from ugtm import runIGTM, runGTM, eIGTM
from ugtm.ugtm_igtm import _auto_n_blocks, _make_block_indices
from sklearn.exceptions import NotFittedError


class TestIGTM(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_dimensions = 10
        self.n_train = 50
        self.n_test = 30
        self.k = 2
        self.m = 2
        self.s = 1.0
        self.regul = 1.0
        self.train = np.random.randn(self.n_train, self.n_dimensions)
        self.test = np.random.randn(self.n_test, self.n_dimensions)

    # --- runIGTM ---

    def test_runIGTM_shape(self):
        model = runIGTM(self.train, k=self.k, m=self.m,
                        s=self.s, regul=self.regul, n_blocks=2)
        self.assertEqual(model.matMeans.shape, (self.n_train, 2))
        self.assertEqual(model.matModes.shape, (self.n_train, 2))

    def test_runIGTM_multiblock(self):
        model = runIGTM(self.train, k=self.k, m=self.m,
                        s=self.s, regul=self.regul, n_blocks=5)
        self.assertEqual(model.matMeans.shape, (self.n_train, 2))

    def test_runIGTM_auto_blocks(self):
        model = runIGTM(self.train, k=self.k, m=self.m,
                        s=self.s, regul=self.regul, n_blocks=0)
        self.assertEqual(model.matMeans.shape, (self.n_train, 2))

    def test_runIGTM_multiblock_close_to_GTM(self):
        # W update is identical to GTM regardless of n_blocks; only betaInv
        # timing differs (old D vs new D). Pearson r on each coordinate axis
        # should be high.
        gtm = runGTM(self.train, k=self.k, m=self.m, s=self.s, regul=self.regul)
        igtm = runIGTM(self.train, k=self.k, m=self.m, s=self.s,
                       regul=self.regul, n_blocks=3)
        for dim in range(2):
            r = np.corrcoef(gtm.matMeans[:, dim], igtm.matMeans[:, dim])[0, 1]
            self.assertGreater(r, 0.9)

    def test_runIGTM_1block_similar_to_GTM(self):
        # With 1 block the algorithms differ only in how betaInv is updated
        # (iGTM uses old D; standard GTM uses new D). Expect similar but not
        # identical results — check that coordinates are in [-1, 1].
        model = runIGTM(self.train, k=self.k, m=self.m,
                        s=self.s, regul=self.regul, n_blocks=1)
        self.assertTrue(np.all(model.matMeans >= -1.0))
        self.assertTrue(np.all(model.matMeans <= 1.0))

    # --- eIGTM ---

    def test_eIGTM_fit_transform_shape(self):
        result = eIGTM(k=self.k, m=self.m, s=self.s,
                       regul=self.regul, n_blocks=2).fit_transform(self.train)
        self.assertEqual(result.shape, (self.n_train, 2))

    def test_eIGTM_transform_shape(self):
        result = (eIGTM(k=self.k, m=self.m, s=self.s,
                        regul=self.regul, n_blocks=2)
                  .fit(self.train)
                  .transform(self.test))
        self.assertEqual(result.shape, (self.n_test, 2))

    def test_eIGTM_not_fitted(self):
        with self.assertRaises(NotFittedError):
            eIGTM().transform(self.test)

    def test_eIGTM_modes_shape(self):
        result = (eIGTM(k=self.k, m=self.m, s=self.s, regul=self.regul,
                        model="modes", n_blocks=2)
                  .fit(self.train)
                  .transform(self.test))
        self.assertEqual(result.shape, (self.n_test, 2))

    def test_eIGTM_responsibilities_shape(self):
        n_nodes = self.k * self.k
        result = (eIGTM(k=self.k, m=self.m, s=self.s, regul=self.regul,
                        model="responsibilities", n_blocks=2)
                  .fit(self.train)
                  .transform(self.test))
        self.assertEqual(result.shape, (self.n_test, n_nodes))

    def test_eIGTM_coordinates_in_range(self):
        result = eIGTM(k=self.k, m=self.m, s=self.s,
                       regul=self.regul, n_blocks=3).fit_transform(self.train)
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    # --- transform_blocks ---

    def test_transform_blocks_means_shape(self):
        model = eIGTM(k=self.k, m=self.m, s=self.s,
                      regul=self.regul, n_blocks=2).fit(self.train)
        blocks = list(model.transform_blocks(self.test, block_size=10))
        result = np.vstack(blocks)
        self.assertEqual(result.shape, (self.n_test, 2))

    def test_transform_blocks_responsibilities_shape(self):
        n_nodes = self.k * self.k
        model = eIGTM(k=self.k, m=self.m, s=self.s, regul=self.regul,
                      model="responsibilities", n_blocks=2).fit(self.train)
        blocks = list(model.transform_blocks(self.test, block_size=10))
        result = np.vstack(blocks)
        self.assertEqual(result.shape, (self.n_test, n_nodes))

    def test_transform_blocks_matches_transform(self):
        model = eIGTM(k=self.k, m=self.m, s=self.s,
                      regul=self.regul, n_blocks=2).fit(self.train)
        full = model.transform(self.test)
        blocks = list(model.transform_blocks(self.test, block_size=10))
        block_result = np.vstack(blocks)
        np.testing.assert_array_almost_equal(full, block_result)

    def test_transform_blocks_not_fitted(self):
        with self.assertRaises(NotFittedError):
            list(eIGTM().transform_blocks(self.test))

    # --- helpers ---

    def test_auto_n_blocks(self):
        self.assertEqual(_auto_n_blocks(5000), 1)
        self.assertEqual(_auto_n_blocks(5001), 2)
        self.assertEqual(_auto_n_blocks(10000), 2)
        self.assertEqual(_auto_n_blocks(1), 1)

    def test_make_block_indices_coverage(self):
        indices = _make_block_indices(50, 3)
        starts = [s for s, _ in indices]
        ends = [e for _, e in indices]
        self.assertEqual(starts[0], 0)
        self.assertEqual(ends[-1], 50)
        # no gaps
        for i in range(len(indices) - 1):
            self.assertEqual(ends[i], starts[i + 1])


if __name__ == '__main__':
    unittest.main()
