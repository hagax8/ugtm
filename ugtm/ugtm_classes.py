"""Defines classes for initial and optimized GTM model.
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT

from __future__ import print_function
import numpy as np


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

