import unittest
import ugtm
import numpy as np
#check matrixdimensions
class TestGTM(unittest.TestCase):
	def setUp(self):
		np.random.seed(0)
		self.n_dimensions=10
		self.n_individuals=100
		self.n_nodes = 4
		self.k = 2
		self.n_rbf_centers = 4
		self.m = 2
		self.data = np.random.randn(self.n_individuals,self.n_dimensions)
		labels=np.random.choice([1,2],size=self.n_individuals)
		activity = np.random.randn(self.n_individuals,1)
	def test_runGTM(self):
		gtm = ugtm.runGTM(data=self.data,k=self.k)
		self.assertEqual(gtm.converged,True)
		self.assertEqual(gtm.matR.shape,(self.n_individuals,self.n_nodes))
		self.assertEqual(sum(gtm.matR[0,:]),1.0)
		self.assertEqual(gtm.matMeans.shape,(self.n_individuals,2))
		self.assertEqual(gtm.matModes.shape,(self.n_individuals,2))
	def test_runkGTM(self):
		gtm = ugtm.runkGTM(data=self.data,k=self.k,doPCA=True,doKernel=True)
		self.assertEqual(gtm.converged,True)
		self.assertEqual(gtm.matR.shape,(self.n_individuals,self.n_nodes))
		self.assertEqual(sum(gtm.matR[0,:]),1.0)
		self.assertEqual(gtm.matMeans.shape,(self.n_individuals,2))
		self.assertEqual(gtm.matModes.shape,(self.n_individuals,2))
	def test_transform(self):
		gtm = ugtm.runGTM(data=self.data,k=self.k,doPCA=True)
		transformed=ugtm.transform(optimizedModel=gtm,train=self.data,test=self.data,doPCA=True)
		self.assertEqual(transformed.converged,True)
		np.testing.assert_almost_equal(gtm.matR,transformed.matR,decimal=7)
		np.testing.assert_almost_equal(gtm.matMeans,transformed.matMeans,decimal=7)
	
if __name__ == '__main__':
    unittest.main()

