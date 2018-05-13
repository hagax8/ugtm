import unittest
import ugtm
import numpy as np

#check matrixdimensions
class TestGTMWorkflow(unittest.TestCase):
	def test_runWorkflow(self):
		train = np.random.randn(20,10)
		test = np.random.randn(20,10)
		labels=np.random.choice(["class1","class2"],size=20)
		activity = np.random.randn(20,1)
		gtm = ugtm.runGTM(train)
		gtm.write("tests/output_tests/testout1")
		gtm_coordinates = gtm.matMeans
		gtm_modes = gtm.matModes
		gtm_responsibilities = gtm.matR
		gtm.plot_multipanel(output="tests/output_tests/testout2",labels=labels,discrete=True,pointsize=20)
		gtm.plot_multipanel(output="tests/output_tests/testout3",labels=activity,discrete=False,pointsize=20)
		gtm.plot_multipanel(output="tests/output_tests/testout4",labels=labels,discrete=True,pointsize=20,do_interpolate=False)
		gtm.plot_multipanel(output="tests/output_tests/testout5",labels=activity,discrete=False,pointsize=20,do_interpolate=False)
		gtm.plot(output="tests/output_tests/testout6",pointsize=20)
		gtm.plot(output="tests/output_tests/testout7",labels=labels,discrete=True,pointsize=20)
		gtm.plot(output="tests/output_tests/testout8",labels=activity,discrete=False,pointsize=20)
		gtm.plot_html(output="tests/output_tests/testout9",pointsize=20)
		gtm.plot_html(output="tests/output_tests/testout10",labels=activity,discrete=False,pointsize=20)
		gtm.plot_html(output="tests/output_tests/testout11",labels=labels,discrete=True,pointsize=20)
		gtm.plot_html(output="tests/output_tests/testout12",labels=activity,discrete=False,pointsize=20,do_interpolate=False,ids=labels)
		gtm.plot_html(output="tests/output_tests/testout13",labels=labels,discrete=True,pointsize=20,do_interpolate=False)
		transformed=ugtm.transform(optimizedModel=gtm,train=train,test=test)
		transformed.plot_html(output="tests/output_tests/testout14",pointsize=20)
		transformed.plot(output="tests/output_tests/testout15",pointsize=20)
		gtm.plot_html_projection(output="tests/output_tests/testout16",projections=transformed,labels=labels,discrete=True,pointsize=20)
		predicted_labels=ugtm.GTC(train=train,test=test,labels=labels)
		predicted_model=ugtm.advancedGTC(train=train,test=test,labels=labels)
		ugtm.printClassPredictions(predicted_model,"tests/output_tests/testout17")
		predicted=ugtm.GTR(train=train,test=test,labels=activity)
		predicted=ugtm.GTC(train=train,test=test,labels=labels)
		ugtm.crossvalidateGTC(data=train,labels=labels,s=1,l=1,n_repetitions=10,n_folds=5)
		ugtm.crossvalidateGTR(data=train,labels=activity,s=1,l=1)
		ugtm.crossvalidatePCAC(data=train,labels=labels,n_neighbors=7)
		ugtm.crossvalidateSVCrbf(data=train,labels=labels,C=1,gamma=1)
		ugtm.crossvalidateSVCrbf(data=train,labels=labels,C=1)
		ugtm.crossvalidateSVR(data=train,labels=activity,C=1,epsilon=1)
		ugtm.crossvalidatePCAR(data=train,labels=activity,n_neighbors=7)
		gtm = ugtm.runkGTM(train, doKernel=True, kernel="linear")	
if __name__ == '__main__':
    unittest.main(warnings='ignore')
    

