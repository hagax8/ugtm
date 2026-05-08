import unittest
import ugtm
import numpy as np

# check matrixdimensions


class TestGTMWorkflow(unittest.TestCase):
    def test_runWorkflow(self):
        train = np.random.randn(20, 10)
        test = np.random.randn(20, 10)
        labels = np.random.choice(["class1", "class2"], size=20)
        activity = np.random.randn(20)
        gtm = ugtm.runGTM(train)
        gtm.write("tests/output_tests/testout1")
        gtm_coordinates = gtm.matMeans
        gtm_modes = gtm.matModes
        gtm_responsibilities = gtm.matR
        predicted_labels = ugtm.GTC(train=train, test=test, labels=labels)
        predicted_model = ugtm.advancedGTC(
            train=train, test=test, labels=labels)
        predicted_model = ugtm.advancedGTC(
            train=train, test=test, labels=labels, doPCA=True,
            n_components=-1)
        ugtm.printClassPredictions(
            predicted_model, "tests/output_tests/testout17")
        predicted = ugtm.GTR(train=train, test=test, labels=activity)
        predicted = ugtm.GTC(train=train, test=test, labels=labels)
        ugtm.crossvalidateGTC(data=train, labels=labels,
                              s=1, regul=1, n_repetitions=10, n_folds=5)
        ugtm.crossvalidateGTR(data=train, labels=activity, s=1, regul=1)
        ugtm.crossvalidatePCAC(data=train, labels=labels, n_neighbors=7)
        ugtm.crossvalidateSVCrbf(data=train, labels=labels, C=1, gamma=1)
        ugtm.crossvalidateSVCrbf(data=train, labels=labels, C=1)
        ugtm.crossvalidateSVR(data=train, labels=activity, C=1, epsilon=1)
        ugtm.crossvalidatePCAR(data=train, labels=activity, n_neighbors=7)
        gtm = ugtm.runkGTM(train, doKernel=True, kernel="linear")


if __name__ == '__main__':
    unittest.main(warnings='ignore')
