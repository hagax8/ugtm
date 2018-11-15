""" ugtm plot functions.
"""
# Authors: Helena A. Gaspar <hagax8@gmail.com>
# License: MIT
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
from scipy import interpolate
import matplotlib.pyplot as plt
import mpld3
from . import ugtm_landscape
import json
from mpld3 import _display


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


_display.NumpyEncoder = NumpyEncoder


def plot(coordinates, labels=None, title="", output="output",
         discrete=False, pointsize=1.0, alpha=0.3, cname="Spectral_r",
         output_format="pdf"):
    """ Simple plotting function.

    Parameters
    ----------
    coordinates: array of shape (n_individuals, 2)
        Coordinates to plot.
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
        Output format.

    Returns
    -------
    Image file
        Basic plotting function. Plots coordinates.
    """
    if labels is None:
        colvec = "black"
    elif discrete:
        uniqClasses, label_numeric = np.unique(labels, return_inverse=True)
        colvec = np.squeeze(label_numeric)
    elif not discrete:
        label_numeric = labels
        colvec = np.squeeze(label_numeric)
    fig, ax = plt.subplots()
    ax.grid(color='white', linestyle='solid')
    ax.set_axisbelow(True)
    ax.set_title(title, size=30)
    scatter = ax.scatter(coordinates[:, 0].tolist(),
                         coordinates[:, 1].tolist(),
                         c=colvec, s=20*pointsize, alpha=alpha,
                         cmap=plt.get_cmap(cname),
                         edgecolor='black')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    if not discrete and (labels is not None):
        plt.colorbar(scatter)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])

    if not output_format:
        # Just show the plot, since no output format has been selected
        fig.show()
    else:
        if output_format[0] == ".":
            output_format = output_format[1:]
        output_file = "%s.%s" % (output, output_format)
        fig.savefig(output_file, format=output_format, dpi=500)
        print("\nWrote %s to disk: %s\n" % (output_format, output_file))
    plt.close(fig)


def plot_pdf(*args, **kwargs):
    # This function is here for the sake of backwards compatibility
    kwargs["output_format"] = "pdf"
    return plot(*args, **kwargs)


def plotMultiPanelGTM(optimizedModel, labels, output="output", discrete=False,
                      pointsize=1.0, alpha=0.3, do_interpolate=True,
                      cname="Spectral_r", prior="estimated"):
    """ Multipanel visualization for GTM object - PDF output.

    Parameters
    ----------
    optimizedModel : instance of :class:`~ugtm.ugtm_classes.OptimizedGTM`,
        Optimized GTM model.
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
    if labels is None:
        colvec = "black"
    elif labels is not None and discrete:
        uniqClasses, label_numeric = np.unique(labels, return_inverse=True)
        colvec = np.squeeze(label_numeric)
    elif not discrete:
        label_numeric = labels
        colvec = np.squeeze(label_numeric)
    fig = plt.figure(figsize=(10, 10))
    means = optimizedModel.matMeans
    modes = optimizedModel.matModes
    # plot1: GTM means visualization
    ax = fig.add_subplot(221)
    ax.scatter(means[:, 0], means[:, 1], c=colvec,
               cmap=plt.get_cmap(cname), s=20*pointsize, alpha=alpha,
               edgecolor="black")
    plt.axis('tight')
    plt.xticks([])
    plt.yticks([])
    plt.title('Means')
    # plot2: GTM modes visualization
    ax2 = fig.add_subplot(222)
    ax2.scatter(modes[:, 0], modes[:, 1], c=colvec,
                cmap=plt.get_cmap(cname), s=20*pointsize, alpha=alpha,
                edgecolor="black")
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title('Modes')
    # plot3: GTM landscape visualization
    ax3 = fig.add_subplot(223)
    # if it's label data, the landscape is a class map;
    # otherwise, it is a continuous landscape
    if labels is not None and discrete:
        plotClassMapNoPoints(optimizedModel, label_numeric,
                             do_interpolate=do_interpolate, cname=cname,
                             prior=prior)
    elif labels is not None:
        plotLandscapeNoPoints(optimizedModel, label_numeric,
                              do_interpolate=do_interpolate, cname=cname)
    ax3.scatter(means[:, 0], means[:, 1], c=colvec,
                cmap=plt.get_cmap(cname),
                s=20*pointsize, alpha=alpha, edgecolor="black")
    plt.axis('tight')
    plt.xticks([])
    plt.yticks([])
    # add mapping from mean positions to modes
    # (GTM nodes where the data points have max probability)
    for i in range(means.shape[0]):
        plt.plot([means[i, 0], modes[i, 0]],
                 [means[i, 1], modes[i, 1]], color='grey', linewidth=0.5)

    # plot4: GTM landscape visualization without means/modes mappings
    fig.add_subplot(224)
    if discrete and labels is not None:
        plotClassMapNoPoints(optimizedModel, label_numeric,
                             do_interpolate=do_interpolate, cname=cname,
                             prior=prior)
    elif labels is not None:
        plotLandscapeNoPoints(optimizedModel, label_numeric,
                              do_interpolate=do_interpolate, cname=cname)
    fig.set_size_inches(16, 13)
    fig.savefig(output+".pdf", format='pdf', dpi=500)
    plt.close(fig)
    print("\nWrote pdf to disk: %s\n" % (output+".pdf"))


def plot_html(coordinates, labels=None, ids=None, title="plot",
              discrete=False, output="output", pointsize=1.0, alpha=0.3,
              cname="Spectral_r"):
    """ Simple plotting function: HTML output.

    Parameters
    ----------
    coordinates: array of shape (n_individuals, 2)
        Coordinates to plot.
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

    Returns
    -------
    Image file
        Plots coordinates to HTML file.
    """
    if labels is None:
        colvec = "black"
    elif discrete:
        uniqClasses, label_numeric = np.unique(labels, return_inverse=True)
        colvec = np.squeeze(label_numeric)
    elif not discrete:
        label_numeric = labels
        colvec = np.squeeze(label_numeric)
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
    ax.grid(color='white', linestyle='solid')
    ax.set_title(title, size=30)
    pointsidx = ['point {0}'.format(i + 1)
                 for i in range(coordinates.shape[0])]
    scatter = ax.scatter(coordinates[:, 0].tolist(),
                         coordinates[:, 1].tolist(),
                         c=colvec, s=20*pointsize, alpha=alpha,
                         cmap=plt.get_cmap(cname),
                         edgecolor='black')
    if ids is not None and labels is not None:
        tooltipstr = ["%s: label=%s" % t for t in zip(ids, labels)]
    elif ids is not None:
        tooltipstr = ["%s: id=%s" % t for t in zip(pointsidx, ids)]
    elif labels is not None:
        tooltipstr = ["%s: label=%s" % t for t in zip(pointsidx, labels)]
    else:
        tooltipstr = list(pointsidx)
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=tooltipstr)
    mpld3.plugins.connect(fig, tooltip)
    mpld3.save_html(fig, output+".html")
    plt.close(fig)
    print("\nWrote html plot to disk: %s\n" % (output+".html"))


def plot_html_GTM(optimizedModel, labels=None, ids=None, plot_arrows=True,
                  title="GTM", discrete=False, output="output",
                  pointsize=1.0, alpha=0.3, do_interpolate=True,
                  cname="Spectral_r", prior="estimated"):
    """ Plotting function for GTM object - HTML output.

    Parameters
    ----------
    optimizedModel : instance of :class:`ugtm.ugtm_classes.OptimizedGTM`
        OptimizedGTM model.
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
    if labels is None:
        colvec = "black"
    elif discrete:
        uniqClasses, label_numeric = np.unique(labels, return_inverse=True)
        colvec = np.squeeze(label_numeric)
    elif not discrete:
        label_numeric = labels
        colvec = np.squeeze(label_numeric)
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
    if discrete and labels is not None:
        plotClassMapNoPoints(optimizedModel, label_numeric, prior=prior,
                             do_interpolate=do_interpolate, cname=cname)
    elif labels is not None:
        plotLandscapeNoPoints(optimizedModel, label_numeric,
                              do_interpolate=do_interpolate, cname=cname)
    means = optimizedModel.matMeans
    modes = optimizedModel.matModes
    ax.grid(color='white', linestyle='solid')
    ax.set_title(title, size=30)
    pointsidx = ['point {0}'.format(i + 1) for i in range(means.shape[0])]
    scatter = ax.scatter(means[:, 0], means[:, 1], c=colvec,
                         s=20*pointsize,
                         alpha=alpha, cmap=plt.get_cmap(cname),
                         edgecolor='black')
    if plot_arrows:
        for i in range(means.shape[0]):
            plt.plot([means[i, 0], modes[i, 0]],
                     [means[i, 1], modes[i, 1]], color='grey', linewidth=0.5)
    if ids is not None and labels is not None:
        tooltipstr = ["%s: label=%s" % t for t in zip(ids, labels)]
    elif ids is not None:
        tooltipstr = ["%s: id=%s" % t for t in zip(pointsidx, ids)]
    elif labels is not None:
        tooltipstr = ["%s: label=%s" % t for t in zip(pointsidx, labels)]
    else:
        tooltipstr = list(pointsidx)
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=tooltipstr)
    mpld3.plugins.connect(fig, tooltip)
    mpld3.save_html(fig, output+".html")
    plt.close(fig)
    print("\nWrote html plot to disk: %s\n" % (output+".html"))


def plot_html_GTM_projection(optimizedModel, projections, labels=None,
                             ids=None, plot_arrows=True,
                             title="GTM_projection",
                             discrete=False, output="output",
                             pointsize=1, alpha=0.3,
                             do_interpolate=True, cname="Spectral_r",
                             prior="estimated"):
    """ Returns a GTM landscape with projected data points - HTML output.

    Parameters
    ----------
    optimizedModel : instance of :class:`ugtm.ugtm_classes.OptimizedGTM`
        OptimizedGTM model.
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
    if discrete:
        uniqClasses, label_numeric = np.unique(labels, return_inverse=True)
    elif not discrete:
        label_numeric = labels
    fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
    if discrete and labels is not None:
        plotClassMap(optimizedModel, np.squeeze(label_numeric),
                     do_interpolate=do_interpolate, cname=cname,
                     pointsize=pointsize, alpha=alpha, prior=prior)
    elif labels is not None:
        plotLandscape(optimizedModel, np.squeeze(label_numeric),
                      do_interpolate=do_interpolate, cname=cname,
                      pointsize=pointsize, alpha=alpha)
    means = projections.matMeans
    modes = projections.matModes
    ax.grid(color='white', linestyle='solid')
    ax.set_title(title, size=30)
    pointsidx = ['point {0}'.format(i + 1) for i in range(means.shape[0])]
    scatter = ax.scatter(means[:, 0], means[:, 1], c="black",
                         s=20*pointsize, alpha=alpha, edgecolor='black')
    if plot_arrows:
        for i in range(means.shape[0]):
            plt.plot([means[i, 0], modes[i, 0]],
                     [means[i, 1], modes[i, 1]], color='grey', linewidth=0.5)
    if ids is not None:
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=list(ids))
    else:
        tooltip = mpld3.plugins.PointLabelTooltip(
            scatter, labels=list(pointsidx))
    mpld3.plugins.connect(fig, tooltip)
    mpld3.save_html(fig, output+".html")
    plt.close(fig)
    print("\nWrote html plot to disk: %s\n" % (output+".html"))


def plotLandscape(optimizedModel, labels, do_interpolate=True,
                  cname="Spectral_r", pointsize=1.0, alpha=0.3):
    """ Plots GTM landscape. Internal usage.
    """
    k = np.sqrt(optimizedModel.matX.shape[0])
    n = 100
    x = optimizedModel.matX[:, 0]
    y = optimizedModel.matX[:, 1]
    z = ugtm_landscape.landscape(optimizedModel, labels)
    if do_interpolate:
        ti = np.linspace(-1.1, 1.1, n)
        XI, YI = np.meshgrid(ti, ti)
        f = interpolate.NearestNDInterpolator(optimizedModel.matX, z)
        ZI = f(XI, YI)
        plt.pcolor(XI, YI, ZI, cmap=plt.get_cmap(cname))
    else:
        plt.scatter(x, y, 50*(10/k), z, cmap=plt.get_cmap(cname), marker="s")
    plt.scatter(optimizedModel.matMeans[:, 0], optimizedModel.matMeans[:, 1],
                s=pointsize*20, c=np.squeeze(labels), cmap=plt.get_cmap(cname),
                edgecolor='black', marker="o")
    plt.title('Landscape')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.colorbar()
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])


def plotLandscapeNoPoints(optimizedModel, labels, do_interpolate=True,
                          cname="Spectral_r"):
    """ Plots GTM landscape without 2D representations. Internal usage.
    """
    k = np.sqrt(optimizedModel.matX.shape[0])
    n = 100
    x = optimizedModel.matX[:, 0]
    y = optimizedModel.matX[:, 1]
    z = ugtm_landscape.landscape(optimizedModel, labels)
    if do_interpolate:
        ti = np.linspace(-1.1, 1.1, n)
        XI, YI = np.meshgrid(ti, ti)
        f = interpolate.NearestNDInterpolator(optimizedModel.matX, z)
        ZI = f(XI, YI)
        plt.pcolor(XI, YI, ZI, cmap=plt.get_cmap(cname))
    else:
        plt.scatter(x, y, 50*(10/k), z, cmap=plt.get_cmap(cname), marker="s")
    plt.title('Landscape')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.colorbar()
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])


def plotClassMap(optimizedModel, labels, prior="estimated",
                 do_interpolate=True, cname="Spectral_r", pointsize=1.0,
                 alpha=0.3):
    """ Plots GTM class map. Internal usage.
    """
    k = np.sqrt(optimizedModel.matX.shape[0])
    n = 100
    x = optimizedModel.matX[:, 0]
    y = optimizedModel.matX[:, 1]
    uniqClasses, label_numeric = np.unique(labels, return_inverse=True)
    z = ugtm_landscape.classMap(optimizedModel,
                                label_numeric, prior).activityModel
    if do_interpolate:
        ti = np.linspace(-1.1, 1.1, n)
        XI, YI = np.meshgrid(ti, ti)
        f = interpolate.NearestNDInterpolator(optimizedModel.matX, z)
        ZI = f(XI, YI)
        plt.pcolor(XI, YI, ZI, cmap=plt.get_cmap(cname))
    else:
        plt.scatter(x, y, 175*(10/k), z, cmap=plt.get_cmap(cname),
                    marker="s", alpha=0.3)
    plt.scatter(optimizedModel.matMeans[:, 0], optimizedModel.matMeans[:, 1],
                s=20*pointsize, c=np.squeeze(label_numeric),
                cmap=plt.get_cmap(cname),
                edgecolor='black', marker="o")
    plt.title('Class Map')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])


def plotClassMapNoPoints(optimizedModel, labels, prior="estimated",
                         do_interpolate=True, cname="Spectral_r"):
    """Plots GTM class map without 2D representations. Internal usage.
    """
    k = np.sqrt(optimizedModel.matX.shape[0])
    n = 100
    x = optimizedModel.matX[:, 0]
    y = optimizedModel.matX[:, 1]
    uniqClasses, label_numeric = np.unique(labels, return_inverse=True)
    z = ugtm_landscape.classMap(optimizedModel,
                                label_numeric, prior).activityModel
    if do_interpolate:
        ti = np.linspace(-1.1, 1.1, n)
        XI, YI = np.meshgrid(ti, ti)
        f = interpolate.NearestNDInterpolator(optimizedModel.matX, z)
        ZI = f(XI, YI)
        plt.pcolor(XI, YI, ZI, cmap=plt.get_cmap(cname))
    else:
        plt.scatter(x, y, 175*(10/k), z, cmap=plt.get_cmap(cname),
                    marker="s", alpha=0.3)
    plt.title('Class Map')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
