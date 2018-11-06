import unittest
import ugtm
import numpy as np


class TestGTM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n_dimensions = 10
        self.n_individuals = 100
        self.n_nodes = 4
        self.k = 2
        self.n_rbf_centers = 4
        self.m = 2
        self.data = np.random.randn(self.n_individuals, self.n_dimensions)
        self.labels = np.random.choice([1, 2], size=self.n_individuals)
        self.gtm = ugtm.runGTM(self.data)

    def test_plotClassMap_html(self):
        self.gtm.plot_html(output="tests/output_tests/testout",
                           discrete=True, pointsize=20, cname="Spectral_r",
                           do_interpolate=False,
                           prior="equiprobable",
                           labels=self.labels)
        self.gtm.plot_html(output="tests/output_tests/testout",
                           labels=self.labels,
                           discrete=True, pointsize=20, cname="Spectral_r",
                           do_interpolate=True, prior="estimated")

    def test_plotClassMap_pdf(self):
        self.gtm.plot(output="tests/output_tests/testout", labels=self.labels,
                      discrete=True, pointsize=20, cname="Spectral_r",
                      )
        self.gtm.plot(output="tests/output_tests/testout", labels=self.labels,
                      discrete=True, pointsize=20, cname="Spectral_r",
                      )

    def test_plotClassMap_multipanel(self):
        self.gtm.plot_multipanel(output="tests/output_tests/testout",
                                 labels=self.labels,
                                 discrete=True,
                                 pointsize=20, cname="Spectral_r",
                                 do_interpolate=False, prior="estimated")
        self.gtm.plot_multipanel(output="tests/output_tests/testout",
                                 labels=self.labels,
                                 discrete=True,
                                 pointsize=20, cname="Spectral_r",
                                 do_interpolate=True, prior="estimated")

    def test_plotLandscape_html(self):
        self.gtm.plot_html(output="tests/output_tests/testout",
                           labels=self.labels,
                           discrete=False, pointsize=20,
                           cname="Spectral_r",
                           do_interpolate=False)
        self.gtm.plot_html(output="tests/output_tests/testout",
                           labels=self.labels,
                           cname="Spectral_r",
                           do_interpolate=True,
                           discrete=False, pointsize=20)

    def test_plotLandscape_pdf(self):
        self.gtm.plot_html(output="tests/output_tests/testout",
                           labels=self.labels,
                           discrete=False, pointsize=20,
                           cname="Spectral_r",
                           do_interpolate=False)
        self.gtm.plot_html(output="tests/output_tests/testout",
                           labels=self.labels,
                           cname="Spectral_r",
                           do_interpolate=True,
                           discrete=False, pointsize=20)

    def test_plotLandscape_multipanel(self):
        self.gtm.plot_multipanel(output="tests/output_tests/testout",
                                 labels=self.labels,
                                 discrete=False, pointsize=20,
                                 cname="Spectral_r",
                                 do_interpolate=False,
                                 )
        self.gtm.plot_multipanel(output="tests/output_tests/testout",
                                 labels=self.labels,
                                 cname="Spectral_r",
                                 discrete=False,
                                 do_interpolate=True,
                                 pointsize=20)


if __name__ == '__main__':
    unittest.main()
