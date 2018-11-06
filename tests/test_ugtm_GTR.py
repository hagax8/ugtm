import unittest
import ugtm
import numpy as np


class TestGTM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n_dimensions = 10
        self.n_train = 50
        self.n_test = 60
        self.n_nodes = 4
        self.k = 2
        self.m = 2
        self.s = 1
        self.l = 1
        self.n_neighbors = 7
        self.train = np.random.randn(self.n_train, self.n_dimensions)
        self.test = np.random.randn(self.n_test, self.n_dimensions)
        self.labels = np.random.randn(self.n_train, 1)

    def test_GTR(self):
        ugtm.GTR(train=self.train, test=self.test, labels=self.labels)

    def test_crossvalGTR(self):
        ugtm.crossvalidateGTR(data=self.train, labels=self.labels,
                              s=self.s, l=self.l, n_repetitions=10, n_folds=5)

    def test_crossvalPCA(self):
        ugtm.crossvalidatePCAR(data=self.train,
                               labels=self.labels,
                               n_neighbors=self.n_neighbors)

    def test_crossvalSVM(self):
        ugtm.crossvalidateSVR(data=self.train, labels=self.labels,
                              n_repetitions=2, n_folds=2, C=1, epsilon=0.1)


if __name__ == '__main__':
    unittest.main()
