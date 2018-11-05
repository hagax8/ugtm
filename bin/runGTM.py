import ugtm
import numpy as np
import time
import argparse
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn import manifold
import math


# argument parsing
parser = argparse.ArgumentParser(description='Generate and assess GTM maps '
                                             'for classification '
                                             'or regression.')
parser.add_argument('--data',
                    help='data file in csv format without header '
                         '(must be a similarity matrix '
                         'if --model kGTM is set, otherwise not)',
                    dest='filenamedat')
parser.add_argument('--labels',
                    help='label file in csv format without header',
                    dest='filenamelbls')
parser.add_argument('--ids',
                    help='label file in csv format without header',
                    dest='filenameids')
parser.add_argument('--testids',
                    help='label file in csv format without header',
                    dest='filenametestids')
parser.add_argument('--labeltype',
                    help='labels are continuous or discrete',
                    dest='labeltype',
                    choices=['continuous', 'discrete'])
parser.add_argument('--usetest',
                    help='use S or swiss roll or iris test data',
                    dest='usetest',
                    choices=['s', 'swiss', 'iris'])
parser.add_argument('--model',
                    help='GTM model, kernel GTM model, SVM, PCA or '
                         'comparison between: '
                         'GTM, kGTM, LLE and tSNE for simple visualization, '
                         'GTM and SVM for regression or '
                         'classification (--crossvalidate); '
                         'benchmarked parameters for GTM are '
                         'regularization and rbf_width_factor '
                         'for given grid_size and rbf_grid_size',
                    dest='model',
                    choices=['GTM', 'kGTM', 'SVM',
                             'PCA', 't-SNE', 'SVMrbf', 'compare'])
parser.add_argument('--output',
                    help='output name',
                    dest='output')
parser.add_argument('--crossvalidate',
                    help='show best l (regularization coefficient) '
                         'and s (RBF width factor) '
                         'for classification or regression, '
                         'with default grid size parameter '
                         'k = sqrt(5*sqrt(Nfeatures))+2) '
                         'and RBF grid size parameter m = sqrt(k); '
                         'you can also set the 4 parameters '
                         'and run only one model with '
                         '--rbf_width_factor, --regularization, '
                         '--grid_size and --rbf_grid_size',
                    action='store_true')
parser.add_argument('--pca',
                    help='do PCA preprocessing; if --n_components is not set, '
                         'will use number of PCs explaining 80%% of variance',
                    action='store_true')
parser.add_argument('--missing',
                    help='there is missing data (encoded by NA)',
                    action='store_true')
parser.add_argument('--test',
                    help='test data; only available for '
                         'GTM classification in this script '
                         '(define training data with --data and '
                         'training labels with --labels '
                         'with --labeltype discrete)',
                    dest='test')
parser.add_argument('--missing_strategy',
                    help='missing data strategy, '
                         'default is median',
                    const='median',
                    type=str,
                    default='median',
                    nargs='?',
                    dest='missing_strategy',
                    choices=['mean', 'median', 'most_frequent'])
parser.add_argument('--predict_mode',
                    help='predict mode for GTM classification: '
                         'default is bayes for an equiprobable '
                         'class prediction, '
                         'you can change this to knn; '
                         'knn is the only one available '
                         'for PCA and t-SNE, '
                         'this option is only useful for GTM',
                    const='bayes',
                    type=str,
                    default='bayes',
                    nargs='?',
                    dest='predict_mode',
                    choices=['bayes', 'knn'])
parser.add_argument('--prior',
                    help='type of prior for GTM classification map '
                         'and prediction model: '
                         'you can choose equiprobable classes '
                         '(prior any class=1/nClasses) '
                         'or to estimate classes from the training set '
                         '(prior class 1 = '
                         'sum(class 1 instances in train)/sum(instances '
                         'in train))',
                    const='equiprobable',
                    type=str,
                    default='equiprobable',
                    nargs='?',
                    dest='prior',
                    choices=['equiprobable', 'estimated'])
parser.add_argument('--n_components',
                    help='set number of components for PCA pre-processing, '
                         'if --pca flag is present',
                    const=-1,
                    type=int,
                    default=-1,
                    nargs='?',
                    dest='n_components')
parser.add_argument('--percentage_components',
                    help='set number of components for PCA pre-processing, '
                         'if --pca flag is present',
                    const=0.80,
                    type=float,
                    default=0.80,
                    nargs='?',
                    dest='n_components')
parser.add_argument('--regularization',
                    help='set regularization factor, default: 0.1; '
                         'set this to -1 to crossvalidate '
                         'when using --crossvalidate',
                    type=float,
                    dest='regularization',
                    default=0.1,
                    nargs='?',
                    const=-1.0)
parser.add_argument('--rbf_width_factor',
                    help='set RBF (radial basis function) width factor, '
                         'default: 0.3; '
                         'set this to -1 to crossvalidate '
                         'when using --crossvalidate',
                    type=float,
                    dest='rbf_width_factor',
                    default=0.3,
                    nargs='?',
                    const=0.3)
parser.add_argument('--svm_margin',
                    help='set C parameter for SVC or SVR',
                    const=1.0,
                    type=float,
                    default=1.0,
                    nargs='?',
                    dest='svm_margin')
parser.add_argument('--svm_epsilon',
                    help='set svr epsilon parameter',
                    const=1.0,
                    type=float,
                    default=1.0,
                    nargs='?',
                    dest='svm_epsilon')
parser.add_argument('--point_size',
                    help='point size',
                    const=1.0,
                    type=float,
                    default=1.0,
                    nargs='?',
                    dest='pointsize')
parser.add_argument('--alpha',
                    help='alpha for scatter plots',
                    const=0.5,
                    type=float,
                    default=0.5,
                    nargs='?',
                    dest='alpha')
parser.add_argument('--svm_gamma',
                    help='set gamma parameter for SVM',
                    const=1.0,
                    type=float,
                    default=1.0,
                    nargs='?',
                    dest='svm_gamma')
parser.add_argument('--grid_size',
                    help='grid size (if k: the map will be kxk, '
                         'default k = sqrt(5*sqrt(Nfeatures))+2)',
                    type=int,
                    dest='grid_size',
                    default=0)
parser.add_argument('--rbf_grid_size',
                    help='RBF grid size (if m: the RBF grid will be mxm, '
                    'default m = sqrt(grid_size))',
                    type=int,
                    dest='rbf_grid_size',
                    default=0)
parser.add_argument('--n_neighbors',
                    help='set number of neighbors for predictive modelling',
                    const=1,
                    type=int,
                    default=1,
                    nargs='?',
                    dest='n_neighbors')
parser.add_argument('--random_state',
                    help='change random state for map initialization '
                         '(default is 5)',
                    const=1234,
                    type=int,
                    default=1234,
                    nargs='?',
                    dest='random_state')
parser.add_argument('--representation',
                    help='type of representation used for GTM: '
                         'modes or means',
                    dest='representation',
                    const='modes',
                    type=str,
                    default='modes',
                    nargs='?',
                    choices=['means', 'modes'])
parser.add_argument('--kernel',
                    help='type of kernel for Kernel GTM - '
                         'default is euclidean',
                    dest='kernel',
                    const='euclidean',
                    type=str,
                    default='euclidean',
                    nargs='?',
                    choices=['euclidean', 'laplacian',
                             'jaccard', 'cosine', 'linear'])
parser.add_argument('--cmap',
                    help='matplotlib color map - '
                         'default is Spectral_r',
                    dest='cname',
                    const='Spectral_r',
                    type=str,
                    default='Spectral_r',
                    nargs='?',
                    choices=['Greys', 'Purples', 'Blues', 'Greens',
                             'Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd',
                             'afmhot', 'autumn', 'bone', 'cool', 'copper',
                             'gist_heat', 'gray', 'hot', 'pink',
                             'spring', 'summer', 'winter',
                             'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic',
                             'Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3',
                             'gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism',
                             'Greys_r', 'Purples_r', 'Blues_r', 'Greens_r',
                             'Blues_r', 'BuGn_r', 'BuPu_r',
                             'GnBu_r', 'Greens_r', 'Greys_r', 'Oranges_r',
                             'OrRd_r',
                             'PuBu_r', 'PuBuGn_r', 'PuRd_r', 'Purples_r',
                             'RdPu_r',
                             'Reds_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r',
                             'YlOrRd_r',
                             'afmhot_r', 'autumn_r', 'bone_r', 'cool_r',
                             'copper_r',
                             'gist_heat_r', 'gray_r', 'hot_r', 'pink_r',
                             'spring_r', 'summer_r', 'winter_r',
                             'BrBG_r', 'bwr_r', 'coolwarm_r', 'PiYG_r',
                             'PRGn_r', 'PuOr',
                             'RdBu_r', 'RdGy_r', 'RdYlBu_r', 'RdYlGn_r',
                             'Spectral_r',
                             'seismic_r',
                             'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r',
                             'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r',
                             'gist_earth_r', 'terrain_r', 'ocean_r',
                             'gist_stern_r',
                             'brg_r', 'CMRmap_r', 'cubehelix_r',
                             'gnuplot_r', 'gnuplot2_r', 'gist_ncar_r',
                             'nipy_spectral_r', 'jet_r', 'rainbow_r',
                             'gist_rainbow_r', 'hsv_r', 'flag_r', 'prism_r'])
parser.add_argument('--verbose',
                    help='verbose mode',
                    action='store_true')
parser.add_argument('--interpolate',
                    help='interpolate between GTM nodes in visualizations',
                    action='store_true')


args = parser.parse_args()
print('')
print(args)
print('')


# process some of the arguments; make sure data is preprocessed if model is PCA
if args.model == 'PCA':
    args.pca = True
    args.n_components = 2
discrete = False
if args.labeltype == "discrete":
    discrete = True

if args.model and ((args.filenamedat and args.filenamelbls)):
    print("User provided model, data file and label names.")
    print("")
elif args.model and (args.usetest):
    print("User provided model and chose a default dataset.")
    print("")
else:
    print("Please provide model and data + labels or model + test data.")
    print("")
    exit

labels = None
data = None
ids = None

# load test examples if we choose to use default data from sklearn
if args.usetest == 's':
    data, labels = sklearn.datasets.samples_generator.make_s_curve(
        500, random_state=args.random_state)
#    ids = np.copy(labels)
elif args.usetest == 'swiss':
    data, labels = sklearn.datasets.make_swiss_roll(
        n_samples=500, random_state=args.random_state)
#    ids = np.copy(labels)
elif args.usetest == 'iris':
    iris = sklearn.datasets.load_iris()
    data = iris.data
    labels = iris.target_names[iris.target]
    discrete = True
#    ids = np.copy(labels)
# if it's not test data, then load provided data files
elif args.filenamedat:
    data = np.genfromtxt(args.filenamedat, delimiter=",", dtype=np.float64)

if args.filenamelbls:
    if discrete is True:
        labels = np.genfromtxt(args.filenamelbls, delimiter="\t", dtype=str)
    else:
        labels = np.genfromtxt(args.filenamelbls, delimiter="\t", dtype=float)
#    ids = np.copy(labels)

# load ids for data points if there are provides
if args.filenameids is not None:
    ids = np.genfromtxt(args.filenameids, delimiter="\t", dtype=str)


# define type of experiment
if (args.crossvalidate is True):
    type_of_experiment = 'crossvalidate'
elif (args.test is not None and discrete is True and args.model == 'GTM'):
    type_of_experiment = 'traintest'
else:
    type_of_experiment = 'visualization'


# TYPE OF EXPERIMENT: 1: CROSSVALIDATION: CAN BE SVM, GTM, PCA
###################################################
###################################################
############# CROSSVALIDATION #####################
###################################################
###################################################

if type_of_experiment == 'crossvalidate':
    ugtm.whichExperiment(data, labels, args, discrete)
    exit


# TYPE OF EXPERIMENT: 2: TRAIN/TEST PREDICTION, CLASSIFICATION WITH GTM
###################################################
###################################################
################# TRAINTEST #######################
###################################################
###################################################


# in case it's a train/test experiment for GTM classification
elif type_of_experiment == 'traintest':
    test = np.genfromtxt(args.test, delimiter=",", dtype=np.float64)
    if args.filenametestids is not None:
        testids = np.genfromtxt(args.filenametestids,
                                delimiter="\t", dtype=str)
    else:
        testids = ""
    prediction = ugtm.advancedGTC(train=data, labels=labels,
                                  test=test, doPCA=args.pca,
                                  n_components=args.n_components,
                                  n_neighbors=args.n_neighbors,
                                  representation=args.representation,
                                  missing=args.missing,
                                  missing_strategy=args.missing_strategy,
                                  random_state=args.random_state,
                                  k=args.grid_size, m=args.rbf_grid_size,
                                  predict_mode=args.predict_mode,
                                  prior=args.prior, l=args.regularization,
                                  s=args.rbf_width_factor)
    prediction['optimizedModel'].plot_html(ids=ids, plot_arrows=True,
                                           title="GTM",
                                           labels=labels,
                                           discrete=discrete,
                                           output=args.output+"_trainedMap",
                                           cname=args.cname,
                                           pointsize=args.pointsize,
                                           alpha=args.alpha,
                                           prior=args.prior,
                                           do_interpolate=args.interpolate)
    ugtm.printClassPredictions(prediction, output=args.output)
    prediction['optimizedModel'].plot_html_projection(labels=labels,
                                                      projections=prediction["indiv_projections"],
                                                      ids=testids,
                                                      plot_arrows=True,
                                                      title="GTM projection",
                                                      discrete=discrete,
                                                      cname=args.cname,
                                                      pointsize=args.pointsize,
                                                      output=args.output,
                                                      alpha=args.alpha,
                                                      prior=args.prior,
                                                      do_interpolate=args.interpolate)
    exit

# TYPE OF EXPERIMENT: 3: VISUALIZATION: CAN BE t-SNE, GTM, PCA
###################################################
###################################################
########### VISUALIZATION #########################
###################################################
###################################################

elif type_of_experiment == 'visualization':

    if args.model != 'GTM':
        data = ugtm.pcaPreprocess(data=data, doPCA=args.pca,
                                  n_components=args.n_components,
                                  missing=args.missing,
                                  missing_strategy=args.missing_strategy,
                                  random_state=args.random_state)

    # set default parameters
    k = int(math.sqrt(5*math.sqrt(data.shape[0])))+2
    m = int(math.sqrt(k))
    l = 0.1
    s = 0.3
    niter = 1000
    maxdim = 100

    # set parameters if provided in options
    if args.regularization:
        l = args.regularization
    if args.rbf_width_factor:
        s = args.rbf_width_factor
    if args.grid_size:
        k = args.grid_size
    if args.rbf_grid_size:
        m = args.rbf_grid_size

    # PCA visualization
    if args.model == 'PCA':
        # if discrete:
        #    uniqClasses, labels = np.unique(labels, return_inverse=True)
        ugtm.plot_html(labels=labels, coordinates=data, ids=ids,
                       title="", output=args.output, cname=args.cname,
                       pointsize=args.pointsize, alpha=args.alpha,
                       discrete=discrete)
        np.savetxt(args.output+".csv", data[:, 0:2], delimiter=',')
        exit

    # t-SNE visualization
    elif args.model == 't-SNE':
       #     if discrete:
       #         uniqClasses, labels = np.unique(labels, return_inverse=True)
        tsne = manifold.TSNE(n_components=2, init='pca',
                             random_state=args.random_state)
        data_r = tsne.fit_transform(data)
        ugtm.plot_html(labels=labels, coordinates=data_r, ids=ids,
                       discrete=discrete,
                       output=args.output, cname=args.cname,
                       pointsize=args.pointsize, alpha=args.alpha, title="")
        ugtm.plot(labels=labels, coordinates=data_r, discrete=discrete,
                  output=args.output, cname=args.cname,
                  pointsize=args.pointsize, alpha=args.alpha,
                  title="")
        np.savetxt(args.output+".csv", data_r, delimiter=',')
        exit

    # GTM visualization
    elif args.model == 'GTM':
        start = time.time()
        gtm = ugtm.runGTM(data=data, k=k, m=m, s=s, l=l, niter=niter,
                          doPCA=args.pca, n_components=args.n_components,
                          missing=args.missing,
                          missing_strategy=args.missing_strategy,
                          random_state=args.random_state, verbose=args.verbose)
        print("k:%s, m:%s, l:%s, s:%s" % (k, m, l, s))
        end = time.time()
        elapsed = end - start
        print("time taken for GTM: ", elapsed)
        np.savetxt(args.output+"_means.csv", gtm.matMeans, delimiter=',')
        gtm.plot_multipanel(
            labels=labels, output=args.output+"_multipanel", discrete=discrete,
            cname=args.cname, pointsize=args.pointsize, alpha=args.alpha,
            prior=args.prior, do_interpolate=args.interpolate)
        gtm.plot_html(labels=labels, ids=ids,
                      discrete=discrete, output=args.output,
                      cname=args.cname, pointsize=args.pointsize,
                      alpha=args.alpha, title="",
                      prior=args.prior, do_interpolate=args.interpolate)
        gtm.plot(labels=labels, output=args.output, discrete=discrete,
                 pointsize=args.pointsize, alpha=args.alpha,
                 cname=args.cname)
        exit

    # if it's for kGTM visualization
    elif args.model == 'kGTM':
        # kGTM embedding
        print("k:%s, m:%s, l:%s, s:%s" % (k, m, l, s))
        matK = ugtm.chooseKernel(data, args.kernel)
        start = time.time()
        kgtm = ugtm.runkGTM(data=matK, k=k, m=m, s=s, l=l,
                            niter=niter, doKernel=False, maxdim=maxdim,
                            doPCA=False,
                            missing=args.missing,
                            missing_strategy=args.missing_strategy,
                            random_state=args.random_state)
        end = time.time()
        elapsed = end - start
        print("time taken for kGTM: ", elapsed)
        # make pdf
        np.savetxt(args.output+"_means.csv", kgtm.matMeans, delimiter=',')
        kgtm.plot_multipanel(
            labels=labels, output=args.output, discrete=discrete,
            cname=args.cname, pointsize=args.pointsize, alpha=args.alpha,
            prior=args.prior, do_interpolate=args.interpolate)
        # interactive plot
        kgtm.plot_html(labels=labels, ids=ids, plot_arrows=True,
                       discrete=discrete, output=args.output, cname=args.cname,
                       pointsize=args.pointsize, alpha=args.alpha,
                       prior=args.prior, do_interpolate=args.interpolate,
                       title="")
        kgtm.plot(labels=labels, output=args.output, discrete=discrete,
                  pointsize=args.pointsize, alpha=args.alpha,
                  cname=args.cname)
        exit

    # if it's to compare GTM, PCA, LLE and t_SNE visualizations
    elif args.model == 'compare':
        if discrete:
            uniqClasses, labels = np.unique(labels, return_inverse=True)
        print("Computing GTM embedding")
        start = time.time()
        gtm = ugtm.runGTM(data=data, k=k, m=m, s=s, l=l, niter=niter,
                          doPCA=args.pca, n_components=args.n_components,
                          missing=args.missing,
                          missing_strategy=args.missing_strategy,
                          random_state=args.random_state)
        end = time.time()
        elapsed = end - start
        print("time taken for GTM: ", elapsed)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(331)
        ax.scatter(data[:, 0], data[:, 1], c=labels,
                   cmap=plt.get_cmap(args.cname), s=20*args.pointsize,
                   alpha=args.alpha)
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        if args.pca:
            plt.title('PCA')
        else:
            plt.title('Original data')
        ax = fig.add_subplot(334)
        ax.scatter(gtm.matMeans[:, 0], gtm.matMeans[:, 1],
                   alpha=args.alpha, s=20*args.pointsize,
                   c=labels, cmap=plt.get_cmap(args.cname))
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        plt.title('GTM')
        ax = fig.add_subplot(337)
        if discrete:
            ugtm.plotClassMap(gtm, labels, cname=args.cname,
                              pointsize=args.pointsize, alpha=args.alpha,
                              prior=args.prior)
        else:
            ugtm.plotLandscape(gtm, labels, cname=args.cname, alpha=args.alpha,
                               pointsize=args.pointsize)
        matK = ugtm.chooseKernel(data, 'laplacian')
        print("Computing kGTM embedding (laplacian)")
        start = time.time()
        kgtm = ugtm.runkGTM(data=matK, k=k, m=m, s=s, l=l, niter=niter,
                            maxdim=maxdim, doPCA=False, doKernel=False,
                            missing=args.missing,
                            missing_strategy=args.missing_strategy,
                            random_state=args.random_state)
        print("The estimated feature space dimension is: ",
              kgtm.n_dimensions)
        end = time.time()
        elapsed = end - start
        print("time taken for kGTM: ", elapsed)
        ax = fig.add_subplot(335)
        ax.scatter(kgtm.matMeans[:, 0], kgtm.matMeans[:, 1], c=labels,
                   cmap=plt.get_cmap(args.cname), s=20*args.pointsize,
                   alpha=args.alpha)
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        plt.title('kGTM (laplacian)')
        ax = fig.add_subplot(338)
        if discrete:
            ugtm.plotClassMap(kgtm, labels, cname=args.cname, alpha=args.alpha,
                              pointsize=args.pointsize, prior=args.prior)
        else:
            ugtm.plotLandscape(kgtm, labels, cname=args.cname, alpha=args.alpha,
                               pointsize=args.pointsize)
        matK = ugtm.chooseKernel(data, 'euclidean')
        print("Computing kGTM embedding (euclidean)")
        start = time.time()
        kgtm = ugtm.runkGTM(data=matK, k=k, m=m, s=s, l=l, niter=niter,
                            maxdim=maxdim, doKernel=False,
                            missing=args.missing,
                            missing_strategy=args.missing_strategy,
                            random_state=args.random_state)
        print("The estimated feature space dimension is: ",
              kgtm.n_dimensions)
        end = time.time()
        elapsed = end - start
        print("time taken for kGTM: ", elapsed)
        ax = fig.add_subplot(336)
        ax.scatter(kgtm.matMeans[:, 0], kgtm.matMeans[:, 1],
                   c=labels, cmap=plt.get_cmap(args.cname), alpha=args.alpha,
                   s=20*args.pointsize)
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        plt.title('kGTM (euclidean)')
        ax = fig.add_subplot(339)
        if discrete:
            ugtm.plotClassMap(kgtm, labels, cname=args.cname, alpha=args.alpha,
                              pointsize=args.pointsize, prior=args.prior)
        else:
            ugtm.plotLandscape(kgtm, labels, cname=args.cname, alpha=args.alpha,
                               pointsize=args.pointsize)
        print("Computing LLE embedding")
        start = time.time()
        data_r, err = manifold.locally_linear_embedding(
            data, n_neighbors=12, n_components=2)
        end = time.time()
        elapsed = end - start
        print("time taken for LLE: ", elapsed)
        print("Done. Reconstruction error: %g" % err)
        ax = fig.add_subplot(332)
        ax.scatter(data_r[:, 0], data_r[:, 1], c=labels,
                   cmap=plt.get_cmap(args.cname),
                   s=20*args.pointsize, alpha=args.alpha)
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        plt.title('LLE')
        print("Computing t-SNE: embedding")
        start = time.time()
        tsne = manifold.TSNE(n_components=2, init='pca',
                             random_state=args.random_state)
        data_r = tsne.fit_transform(data)
        end = time.time()
        elapsed = end - start
        print("time taken for TSNE: ", elapsed)
        print("Done. Reconstruction error: %g" % err)
        ax = fig.add_subplot(333)
        ax.scatter(data_r[:, 0], data_r[:, 1], c=labels,
                   cmap=plt.get_cmap(args.cname),
                   s=20*args.pointsize, alpha=args.alpha)
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        plt.title('t-SNE')
        fig.savefig(args.output)
        plt.close(fig)
        exit
    else:
        print("Sorry. Model not recognized.")
        exit
else:
    print('Sorry. Could not guess what you wanted. '
          'Remember to define --model '
          'and (--data and --labels) or --model and --usetest.')
    exit
