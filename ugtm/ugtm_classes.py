from __future__ import print_function
import numpy as np
from .ugtm_plot import plot_pdf
from .ugtm_plot import plotMultiPanelGTM
from .ugtm_plot import plot_html_GTM
from .ugtm_plot import plot_html_GTM_projection


class ReturnU(object):
    def __init__(self, matU, betaInv):
        self.matU = matU
        self.betaInv = betaInv


class InitialGTM(object):
    def __init__(self, matX, matM, n_nodes, n_rbf_centers, rbfWidth,
                 matPhiMPlusOne, matW, matY, betaInv, n_dimensions):
        self.matX = matX
        self.matM = matM
        self.n_rbf_centers = n_rbf_centers
        self.n_nodes = n_nodes
        self.rbfWidth = rbfWidth
        self.matPhiMPlusOne = matPhiMPlusOne
        self.matW = matW
        self.matY = matY
        self.betaInv = betaInv
        self.n_dimensions = n_dimensions


class OptimizedGTM(object):
    def __init__(self, matW, matY, matP, matR, betaInv, matMeans,
                 matModes, matX, n_dimensions, converged):
        self.matW = matW
        self.matY = matY
        self.matP = matP
        self.matR = matR
        self.betaInv = betaInv
        self.matMeans = matMeans
        self.matModes = matModes
        self.matX = matX
        self.n_dimensions = n_dimensions
        self.converged = converged

    def write(self, output="output"):
        np.savetxt(fname=output+"_responsibilities.csv",
                   X=self.matR, delimiter=",")
        np.savetxt(fname=output+"_coordinates.csv",
                   X=self.matMeans, delimiter=",")
        np.savetxt(fname=output+"_modes.csv", X=self.matModes, delimiter=",")
        print("")
        print("Wrote to disk:")
        print("")
        print("%s: responsibilities, which represent "
              "each individual's encoding "
              "on the map (dimensions=n_individuals*n_nodes_on_the_map)"
              % (output+"_responsibilities.csv"))
        print("")
        print("%s: coordinates to plot, which represent each individual's "
              "mean position on the map (dimensions = "
              "n_individuals*n_latent_dimensions)"
              % (output+"_coordinates.csv"))
        print("")
        print("%s: modes positions for each individual on the map "
              "(node with max probability for the individual; "
              "dimensions = n_individuals*n_latent_dimensions)"
              % (output+"_modes.csv"))
        print("")
        print("")

    def write_all(self, output="output"):
        outparams = "n_dimensions:"+str(self.n_dimensions) + \
                    "\n"+"inverse_variance:"+str(self.betaInv)
        np.savetxt(fname=output+"_responsibilities.csv",
                   X=self.matR, delimiter=",")
        np.savetxt(fname=output+"_coordinates.csv",
                   X=self.matMeans, delimiter=",")
        np.savetxt(fname=output+"_modes.csv", X=self.matModes, delimiter=",")
        np.savetxt(fname=output+"_manifold.csv", X=self.matY, delimiter=",")
        np.savetxt(fname=output+"_parametersMatrix.csv",
                   X=self.matW, delimiter=",")
        np.savetxt(fname=output+"_dimensionsAndVariance.csv", X=outparams)
        print("")
        print("Wrote to disk:")
        print("")
        print("%s: responsibilities, which represent "
              "each individual's encoding on the map "
              "(dimensions=n_individuals*n_nodes_on_the_map)"
              % (output+"_responsibilities.csv"))
        print("")
        print("%s: coordinates to plot, which represent each individual's "
              "mean position on the map "
              "(dimensions = n_individuals*n_latent_dimensions)"
              % (output+"_coordinates.csv"))
        print("")
        print("%s: modes positions for each individual on the map "
              "(node with max probability for the individual; "
              "dimensions = n_individuals*n_latent_dimensions)"
              % (output+"_modes.csv"))
        print("")
        print("%s: manifold coordinates in the initial data space "
              "(dimensions: n_data_dimensions*n_points_on_manifold"
              % (output+"_manifold.csv"))
        print("")
        print("%s: parameters matrix"
              % (output+"_parametersMatrix.csv"))
        print("")
        print("%s: initial space and inverse variance"
              % (output+"_dimensionsAndVariance.csv"))
        print("")
        print("")

    def plot(self, labels=None, title="", output="output",
             discrete=False, pointsize=1, alpha=0.3, cname="Spectral_r"):
        plot_pdf(coordinates=self.matMeans, labels=labels, title=title,
                 output=output, discrete=discrete,
                 pointsize=pointsize, alpha=alpha, cname=cname)

    def plot_modes(self, labels=None, title="", output="output",
                   discrete=False, pointsize=1, alpha=0.3, cname="Spectral_r"):
        plot_pdf(coordinates=self.matModes, labels=labels, title=title,
                 output=output, discrete=discrete,
                 pointsize=pointsize, alpha=alpha, cname=cname)

    def plot_html(self, labels=None, ids=None, plot_arrows=True,
                  title="GTM", discrete=False, output="output",
                  pointsize=1.0, alpha=0.3, do_interpolate=True,
                  cname="Spectral_r"):
        plot_html_GTM(optimizedModel=self, labels=labels, ids=ids,
                      plot_arrows=plot_arrows, title=title, discrete=discrete,
                      output=output, pointsize=pointsize, alpha=alpha,
                      do_interpolate=do_interpolate, cname=cname)

    def plot_multipanel(self, labels, output="output", discrete=False,
                        pointsize=1.0, alpha=0.3, do_interpolate=True,
                        cname="Spectral_r"):
        plotMultiPanelGTM(optimizedModel=self, labels=labels, output=output,
                          discrete=discrete, pointsize=pointsize, alpha=alpha,
                          do_interpolate=do_interpolate, cname=cname)

    def plot_html_projection(self, projections, labels=None,
                             ids=None, plot_arrows=True,
                             title="GTM_projection", discrete=False,
                             output="output", pointsize=1.0,
                             alpha=0.3, do_interpolate=True,
                             cname="Spectral_r"):
        plot_html_GTM_projection(optimizedModel=self, projections=projections,
                                 labels=labels, ids=ids,
                                 plot_arrows=plot_arrows,
                                 title=title, discrete=discrete,
                                 output=output,
                                 pointsize=pointsize, alpha=alpha,
                                 do_interpolate=do_interpolate, cname=cname)
