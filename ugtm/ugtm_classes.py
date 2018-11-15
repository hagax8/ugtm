"""Defines classes for initial and optimized GTM model.
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np
from .ugtm_plot import plot
from .ugtm_plot import plotMultiPanelGTM
from .ugtm_plot import plot_html_GTM
from .ugtm_plot import plot_html_GTM_projection


class ReturnU(object):
    def __init__(self, matU, betaInv):
        self.matU = matU
        self.betaInv = betaInv


class InitialGTM(object):
    r"""Class for initial GTM model.

    Arguments
    ----------
    matX : array of shape (n_nodes, 2)
        Coordinates of nodes defining a grid in the 2D space.
    matM : array of shape (n_rbf_centers, 2)
        Coordinates of radial basis function (RBF) centers,
        defining a grid in the 2D space.
    n_nodes : int
        The number of nodes defining a grid in the 2D space.
    n_rbf_centers : int
        The number of radial basis function (RBF) centers.
    rbfWidth : float
        Initial radial basis function (RBF) width.
        This is set to the average of the minimum distance between RBF centers:
        :math:`rbfWidth=\sigma \times average(\mathbf{distances(rbf)}_{min})`,
        where :math:`sigma` is the GTM hyperparameter s.
        NB: if GTM hyperparameter s = 0 (not recommended),
        rbfWidth is set to the maximum distance between RBF centers.
    matPhiMPlusOne: array of shape (n_nodes, n_rbf_centers+1)
        RBF matrix plus one dimension to include a term for bias.
    matW: array of shape (n_dimensions, n_rbf_centers+1)
        Parameter matrix (PCA-initialized).
    matY: array of shape (n_dimensions, n_nodes)
        Manifold in n-dimensional space (projection of matX in data space);
        A point matY[:,i] is a center of Gaussian component in data space.
        :math:`\mathbf{Y}=\mathbf{W}\mathbf{\Phi}^T`
    betaInv: float
        Noise variance parameter for the data distribution.
        Written as :math:`\beta^{-1}` in the original paper.
        Initialized to be the larger between:
        (1) the 3rd eigenvalue of the data covariance matrix,
        (2) half the average distance between Gaussian component centers
        in the data space (matY matrix).
    n_dimensions: int
        Data space dimensionality (number of variables).
    """

    def __init__(self, matX, matM, n_nodes, n_rbf_centers, rbfWidth,
                 matPhiMPlusOne, matW, matY, betaInv, n_dimensions):
        r"""Constructor for InitialGTM class.

        Parameters
        ----------
        matX : array of shape (n_nodes, 2)
            Coordinates of nodes defining a grid in the 2D space.
        matM : array of shape (n_rbf_centers, 2)
            Coordinates of radial basis function (RBF) centers,
            defining a grid in the 2D space.
        n_nodes : int
            The number of nodes defining a grid in the 2D space.
        n_rbf_centers : int
            The number of radial basis function (RBF) centers.
        rbfWidth : float
            Initial radial basis function (RBF) width.
            This is set to the average of the minimum distance between RBF centers:
            :math:`rbfWidth=\sigma \times average(\mathbf{distances(rbf)}_{min})`,
            where :math:`sigma` is the GTM hyperparameter s.
            NB: if GTM hyperparameter s = 0 (not recommended),
            rbfWidth is set to the maximum distance between RBF centers.
        matPhiMPlusOne: array of shape (n_nodes, n_rbf_centers+1)
            RBF matrix plus one dimension to include a term for bias.
        matW: array of shape (n_dimensions, n_rbf_centers+1)
            Parameter matrix (PCA-initialized).
        matY: array of shape (n_dimensions, n_nodes)
            Manifold in n-dimensional space (projection of matX in data space);
            A point matY[:,i] is a Gaussian component center in data space.
            :math:`\mathbf{Y}=\mathbf{W}\mathbf{\Phi}^T`
        betaInv: float
            Noise variance parameter for the data distribution.
            Written as :math:`\beta^{-1}` in the original paper.
            Initialized to be the larger between:
            (1) the 3rd eigenvalue of the data covariance matrix,
            (2) half the average distance between Gaussian component centers
            in the data space (matY matrix).
        n_dimensions: int
            Data space dimensionality (number of variables).
        """
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
    r"""Class for optimized GTM model.

    Attributes
    ----------
    matX : array of shape (n_nodes, 2)
        Coordinates of nodes defining a grid in the 2D space.
    matW : array of shape (n_dimensions, n_rbf_centers+1)
        Parameter matrix (PCA-initialized).
    matY : array of shape (n_dimensions, n_nodes)
        Manifold in n-dimensional space (projection of matX in data space).
        matY = np.dot(matW, np.transpose(matPhiMPlusOne))
    matP : array of shape (n_individuals, n_nodes)
        Data distribution with variance betaInv.
    matR : array of shape (n_individuals, n_nodes)
        Responsibilities (posterior probabilities),
        used to compute data representations:
        means (matMeans) and modes (matModes).
        Responsibilities are the main output of GTM.
        matR[i,:] represents the responsibility vector for an instance i.
        The columns in matR correspond to rows in matX (nodes).
    betaInv: float
        Noise variance parameter for the data distribution.
        Written as :math:`\beta^{-1}` in the original paper.
    matMeans : array of shape (n_individuals, 2)
        Data representation in 2D space: means (most commonly used for GTM).
    matModes : array of shape(n_individuals, 2)
        Data representation in 2D space: modes
        (for each instance, coordinate with highest responsibility).
    n_dimensions : int
        Data space dimensionality (number of variables).
    converged : bool
        True if the model has converged; otherwise False.
    """

    def __init__(self, matW, matY, matP, matR, betaInv, matMeans,
                 matModes, matX, n_dimensions, converged):
        r"""Constructor for OptimizedGTM class.

        Parameters
        ----------
        matX : array of shape (n_nodes, 2)
            Coordinates of nodes defining a grid in the 2D space.
        matW : array of shape (n_dimensions, n_rbf_centers+1)
            Parameter matrix (PCA-initialized).
        matY : array of shape (n_dimensions, n_nodes)
            Manifold in n-dimensional space (projection of matX in data space).
            matY = np.dot(matW, np.transpose(matPhiMPlusOne))
        matP : array of shape (n_individuals, n_nodes)
            Data distribution with variance betaInv.
        matR : array of shape (n_individuals, n_nodes)
            Responsibilities (posterior probabilities),
            used to compute data representations:
            means (matMeans) and modes (matModes).
            Responsibilities are the main output of GTM.
            matR[i,:] represents the responsibility vector for an instance i.
            The columns in matR correspond to rows in matX (nodes).
        betaInv: float
            Noise variance parameter for the data distribution.
            Written as :math:`\beta^{-1}` in the original paper.
        matMeans : array of shape (n_individuals, 2)
            Data representation in 2D space: means (most commonly used for GTM).
        matModes : array of shape(n_individuals, 2)
            Data representation in 2D space: modes
            (for each instance, coordinate with highest responsibility).
        n_dimensions : int
            Data space dimensionality (number of variables).
        converged : bool
            True if the model has converged; otherwise False.
        """
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
        """Write optimized GTM model: means, modes and responsibilities.

        Parameters
        ----------
        output : str, optional (default = 'output')
            Output path.

        Returns
        -------
        CSV files
            Separate files for (1) means (mean position for each data point),
            (2) modes (node with max. responsibility for each data point),
            (3) responsibilities (posterior probabilities for each data point)
        """
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
        """Write optimized GTM model and optimized parameters.

        Parameters
        ----------
        output : str, optional (default = 'output')
            Output path.

        Returns
        -------
        CSV files
            Separate files for (1) means (mean position for each data point),
            (2) modes (node with max. responsibility for each data point),
            (3) responsibilities (posterior probabilities for each data point),
            (4) initial space dimension and data distribution variance,
            (5) manifold coordinates (matY),
            (6) parameter matrix (matW)
        """
        outparams = "n_dimensions:"+str(self.n_dimensions) + \
                    "\n"+"variance:"+str(self.betaInv)
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
        print("%s: initial space and variance"
              % (output+"_dimensionsAndVariance.csv"))
        print("")
        print("")

    def plot(self, labels=None, title="", output="output",
             discrete=False, pointsize=1, alpha=0.3, cname="Spectral_r",
             output_format="pdf"):
        """ Simple plotting function for GTM object.

        Parameters
        ----------
        labels : array of shape (n_individuals,), optional (default = None)
            Data labels.
        title : str, optional (default = '')
            Plot title.
        output : str, optional (default = 'ouptut')
            Output path for plot.
        discrete : bool (default = False)
            Type of label; discrete=True if labels are nominal or binary.
        pointsize : float, optional (default = '1')
            Marker size.
        alpha : float, optional (default = '0.3'),
            Marker transparency.
        cname : str, optional (default = 'Spectral_r'),
            Name of matplotlib color map.
            Cf. https://matplotlib.org/examples/color/colormaps_reference.html
        output_format : {'pdf', "png', 'ps', 'eps', 'svg'}
            Output format for GTM plot.

        Returns
        -------
        Image file
            Basic GTM plotting function. Points mean GTM representation
            for each data point.

        Notes
        -----
        This function plots mean representations only (no landscape nor modes).
        """
        plot(coordinates=self.matMeans, labels=labels, title=title,
             output=output, discrete=discrete,
             pointsize=pointsize, alpha=alpha, cname=cname,
             output_format=output_format)

    def plot_modes(self, labels=None, title="", output="output",
                   discrete=False, pointsize=1, alpha=0.3, cname="Spectral_r",
                   output_format="pdf"):
        """ Simple plotting function for GTM object: plot modes.

        Parameters
        ----------
        labels : array of shape (n_individuals,), optional (default = None)
            Data labels.
        title : str, optional (default = '')
            Plot title.
        output : str, optional (default = 'ouptut')
            Output path for plot.
        discrete : bool (default = False)
            Type of label; discrete=True if labels are nominal or binary.
        pointsize : float, optional (default = '1')
            Marker size.
        alpha : float, optional (default = '0.3')
            Marker transparency.
        cname : str, optional (default = 'Spectral_r')
            Name of matplotlib color map.
        output_format : {'png', 'pdf', 'ps', 'eps', 'svg'}, default = 'pdf'
            Output format for GTM plot.

        Returns
        -------
        Image file
            Plot of GTM modes
            (for each data point, node with highest responsibility).

        Notes
        -----
        This function plots mode representations only (no landscape nor means).
        """
        plot(coordinates=self.matModes, labels=labels, title=title,
             output=output, discrete=discrete,
             pointsize=pointsize, alpha=alpha, cname=cname,
             output_format=output_format)

    def plot_html(self, labels=None, ids=None, plot_arrows=True,
                  title="GTM", discrete=False, output="output",
                  pointsize=1.0, alpha=0.3, do_interpolate=True,
                  cname="Spectral_r", prior="estimated"):
        """ Plotting function for GTM object - HTML output.

        Parameters
        ----------
        labels : array of shape (n_individuals,), optional (default = None)
            Data labels.
        ids : array of shape (n_individuals,), optional (default = None)
            Identifiers for each data point - appears in tooltips.
        title : str, optional (default = '')
            Plot title.
        output : str, optional (default = 'ouptut')
            Output path for plot.
        discrete : bool (default = False)
            Type of label; discrete=True if labels are nominal or binary.
        pointsize : float, optional (default = '1')
            Marker size.
        alpha : float, optional (default = '0.3')
            Marker transparency.
        do_interpolate : bool, optional (default = True)
            Interpolate color between grid nodes.
        cname : str, optional (default = 'Spectral_r')
            Name of matplotlib color map.
        prior : {'estimated','equiprobable'}
            Type of prior used to compute class probabilities on the map.
            Choose equiprobable for equiprobable priors.
            Estimated priors take into account class imbalance.

        Returns
        -------
        HTML file
            HTML file of GTM output (mean coordinates). If labels are provided,
            a landscape (continuous or discrete depending on the labels) is
            drawn in the background. This landscape is computed
            using responsibilities and is indicative of the average activity
            or label value at a given node on the map.

        Notes
        -----
        May be time-consuming for large datasets.
        """
        plot_html_GTM(optimizedModel=self, labels=labels, ids=ids,
                      plot_arrows=plot_arrows, title=title, discrete=discrete,
                      output=output, pointsize=pointsize, alpha=alpha,
                      do_interpolate=do_interpolate, cname=cname,
                      prior=prior)

    def plot_multipanel(self, labels, output="output", discrete=False,
                        pointsize=1.0, alpha=0.3, do_interpolate=True,
                        cname="Spectral_r", prior="estimated"):
        """ Multipanel visualization for GTM object - PDF output.

        Parameters
        ----------
        labels : array of shape (n_individuals,), optional (default = None)
            Data labels.
        output : str, optional (default = 'ouptut')
            Output path for plot.
        discrete : bool (default = False)
            Type of label; discrete=True if labels are nominal or binary.
        pointsize : float, optional (default = '1')
            Marker size.
        alpha : float, optional (default = '0.3')
            Marker transparency.
        do_interpolate : bool, optional (default = True)
            Interpolate color between grid nodes.
        cname : str, optional (default = 'Spectral_r')
            Name of matplotlib color map.
        prior : {'estimated','equiprobable'}
            Type of prior used to compute class probabilities on the map.
            Choose equiprobable for equiprobable priors.
            Estimated priors take into account class imbalance.

        Returns
        -------
        PDF image
            Four plots are returned:
            (1) means,
            (2) modes,
            (3) landscape with means,
            (4) landscape with modes.
        """
        plotMultiPanelGTM(optimizedModel=self, labels=labels, output=output,
                          discrete=discrete, pointsize=pointsize, alpha=alpha,
                          do_interpolate=do_interpolate, cname=cname,
                          prior=prior)

    def plot_html_projection(self, projections, labels=None,
                             ids=None, plot_arrows=True,
                             title="GTM_projection", discrete=False,
                             output="output", pointsize=1.0,
                             alpha=0.3, do_interpolate=True,
                             cname="Spectral_r", prior="estimated"):
        """ Returns a GTM landscape with projected data points - HTML output.

        Parameters
        ----------
        projections : instance of :class:`ugtm.ugtm_classes.OptimizedGTM`
            OptimizedGTM model.
        labels : array of shape (n_train,), optional (default = None)
            Data labels for the training set (not projections).
        ids : array of shape (n_test,), optional (default = None)
            Identifiers for each projected data point - appears in tooltips.
        title : str, optional (default = '')
            Plot title.
        output : str, optional (default = 'ouptut')
            Output path for plot.
        discrete : bool (default = False)
            Type of label; discrete=True if labels are nominal or binary.
        pointsize : float, optional (default = '1')
            Marker size.
        alpha : float, optional (default = '0.3')
            Marker transparency.
        do_interpolate : bool, optional (default = True)
            Interpolate color between grid nodes.
        cname : str, optional (default = 'Spectral_r')
            Name of matplotlib color map.
        prior : {'estimated','equiprobable'}
            Type of prior used to compute class probabilities on the map.
            Choose equiprobable for equiprobable priors.
            Estimated priors take into account class imbalance.

        Returns
        -------
        HTML file
            HTML file of GTM output (mean coordinates).
            This function plots a GTM model (from a training set)
            with projected points
            (= mean positions of projected test data).
            If labels are provided,
            a landscape (continuous or discrete depending on the labels) is
            drawn in the background. This landscape is computed
            using responsibilities and is indicative of the average activity
            or label value at a given node on the map.

        Notes
        -----
        - May be time-consuming for large datasets.
        - The labels correspond to training data (the optimized model).
        - The ids (identifiers) correspond to the test data (projections).
        """
        plot_html_GTM_projection(optimizedModel=self, projections=projections,
                                 labels=labels, ids=ids,
                                 plot_arrows=plot_arrows,
                                 title=title, discrete=discrete,
                                 output=output,
                                 pointsize=pointsize, alpha=alpha,
                                 do_interpolate=do_interpolate, cname=cname,
                                 prior=prior)
