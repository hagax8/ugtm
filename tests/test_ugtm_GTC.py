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
        self.labels = np.random.choice([1, 2], size=self.n_train)
        self.activity = np.random.randn(self.n_test, 1)

    def test_GTC_bayes(self):
        gtc = ugtm.GTC(train=self.train, test=self.test, labels=self.labels)

    def test_GTC_knn(self):
        gtc = ugtm.GTC(train=self.train,
                       test=self.test,
                       labels=self.labels,
                       predict_mode="knn",
                       n_neighbors=self.n_neighbors,
                       representation="means"
                       )

    def test_advancedGTC(self):
        predicted_model = ugtm.advancedGTC(train=self.train, test=self.test,
                                           labels=self.labels)
        ugtm.printClassPredictions(predicted_model,
                                   "tests/output_tests/testout23")

    def test_crossvalGTC(self):
        ugtm.crossvalidateGTC(data=self.train, labels=self.labels,
                              s=self.s, l=self.l, n_repetitions=10, n_folds=5)

    def test_crossvalPCA(self):
        ugtm.crossvalidatePCAC(data=self.train,
                               labels=self.labels,
                               n_neighbors=self.n_neighbors)

    def test_crossvalSVM(self):
        ugtm.crossvalidateSVCrbf(data=self.train, labels=self.labels,
                                 C=1, gamma=1, n_folds=2, n_repetitions=2)
        ugtm.crossvalidateSVC(data=self.train, labels=self.labels, C=1.0,
                              n_folds=2, n_repetitions=2)


if __name__ == '__main__':
    unittest.main()
