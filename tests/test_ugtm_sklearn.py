import unittest
import ugtm
from ugtm import eGTM, eGTC, eGTR, eGTCnn
import numpy as np


class TestGTMSklearn(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n_dimensions = 10
        self.n_train = 50
        self.n_test = 60
        self.n_nodes = 4
        self.k = 2
        self.m = 2
        self.s = 1
        self.regul = 1
        self.n_neighbors = 7
        self.train = np.random.randn(self.n_train, self.n_dimensions)
        self.test = np.random.randn(self.n_test, self.n_dimensions)
        self.labels = np.random.choice([1, 2], size=self.n_train)
        self.activity = np.random.randn(self.n_test)

    def test_eGTM(self):
        transformed = eGTM().fit(self.train).transform(self.test)
        self.assertTrue(transformed.shape == (60, 2))

    def test_eGTM_modes(self):
        transformed = eGTM(model="modes").fit(self.train).transform(self.test)
        self.assertTrue(transformed.shape == (60, 2))

    def test_eGTM_trainequalstest(self):
        transformed = eGTM(model="responsibilities").fit(self.train).transform(
            self.train)
        original = eGTM().fit(self.train).optimizedModel.matR
        np.testing.assert_almost_equal(original, transformed, decimal=7)

    def test_eGTC(self):
        predicted_labels = eGTC().fit(
            self.train, self.labels
        ).predict(self.test)
        self.assertTrue(predicted_labels.shape == (60,))

    def test_eGTCnn(self):
        predicted_labels = eGTCnn().fit(
            self.train, self.labels
        ).predict(self.test)
        self.assertTrue(predicted_labels.shape == (60,))

    def test_eGTR(self):
        predicted_labels = eGTR().fit(
            self.train, self.labels
        ).predict(self.test)
        self.assertTrue(predicted_labels.shape == (60,))


if __name__ == '__main__':
    unittest.main()
