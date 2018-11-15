import unittest
import ugtm
import numpy as np
# check matrixdimensions


class TestGTMkGTM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n_dimensions = 10
        self.n_individuals = 100
        self.n_nodes = 4
        self.k = 2
        self.n_rbf_centers = 4
        self.data = np.random.randn(self.n_individuals, self.n_dimensions)
        labels = np.random.choice([1, 2], size=self.n_individuals)
        activity = np.random.randn(self.n_individuals, 1)

    def test_writeGTM(self):
        gtm = ugtm.runGTM(data=self.data, k=self.k)
        gtm.write("tests/output_tests/testout21")

    def test_writekGTM(self):
        gtm = ugtm.runkGTM(data=self.data, k=self.k,
                           doPCA=True, doKernel=True)
        gtm.write("tests/output_tests/testout22")


if __name__ == '__main__':
    unittest.main()
