import tensorflow as tf

def disagreement(logits_1, logits_2):
  """Disagreement between the predictions of two classifiers."""
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  return tf.cast(preds_1 != preds_2, tf.float32)

def cosine_distance(x, y):
  """Cosine distance between vectors x and y."""
  x_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1))
  x_norm = tf.reshape(x_norm, (-1, 1))
  y_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1))
  y_norm = tf.reshape(y_norm, (-1, 1))
  normalized_x = x / x_norm
  normalized_y = y / y_norm
  return tf.reduce_sum(normalized_x * normalized_y, axis=-1)

"""##CLASSE SGH"""

from sklearn.ensemble import BaseEnsemble
from sklearn.linear_model import SGDClassifier
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.utils.validation import check_X_y

from math import sqrt

from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", message="Maximum number of iteration")

class SGH(BaseEnsemble):
    """
    Self-Generating Hyperplanes (SGH).

    Generates a pool of classifiers which guarantees an Oracle
    accuracy rate of 100% over the training (input) set.
    That is, for each instance in the training set, there is at
    least one classifier in the pool able to correctly label it.
    The generated classifiers are always two-class hyperplanes.


    References
    ----------

    L. I. Kuncheva, A theoretical study on six classi
er fusion
    strategies, IEEE Transactions on
    Pattern Analysis and Machine Intelligence 24 (2) (2002) 281-286.

    M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin,
    On the characterization of the
    oracle for dynamic classi
er selection, in: International
    Joint Conference on Neural Networks,
    IEEE, 2017, pp. 332-339.

    """

    def __init__(self,
                 base_estimator=SGDClassifier,
                 n_estimators=1,
                 correct_classif_label=[]
                 ):

        super(SGH, self).__init__(base_estimator=base_estimator,
                                  n_estimators=1)

        # Pool initially empty
        self.estimators_ = []
        #self.base_estimator_ = base_estimator


    def fit(self, X, y, included_samples=np.array([]), sample_weights=None):
        """
        Populates the SHG ensemble.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        included_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which
            samples in X are to be used for training.
            If all, leave blank.

        sample_weights : array of shape = [n_samples]
            array of float indicating the weight of each sample in X.
            Default is None.

        Returns
        -------
        self

        """

        check_X_y(X, y)
        return self._fit(X, y, included_samples, sample_weights=sample_weights)

    def _fit(self, X, y, included_samples, sample_weights=None):

        # Set base estimator as the Perceptron
            # SGDClassifier(loss="perceptron", eta0=1.e-17,
            #                                  max_iter=1,
            #                                  learning_rate="constant",
            #                                  penalty=None)

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size

        # If there is no indication of which instances to
        # include in the training, include all
        if included_samples.sum() == 0:
            included_samples = np.ones((X.shape[0]), int)

        # Generate pool
        self._generate_pool(X, y, included_samples,
                            sample_weights=sample_weights)

        return self

    def _build_Perceptron(self, X, y, curr_training_samples, centroids, sample_weights):
        """
        Calculates the parameters (weight and bias) of the hyperplane
        placed in the midpoint between the centroids of most distant
        classes in X[curr_training_samples].


        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        curr_training_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which
            samples in X are to be used for placing the hyperplane.

        centroids : array of shape = [n_classes,n_features]
            centroids of each class considering the previous
            distribution of X[curr_training_samples].

        sample_weights : array of shape [n_samples]
            weights used to compute the centroid of
            each class in X[curr_training_samples], as well as
            increase/decrease their margin.


        Returns
        -------

        perc : SGDClassifier
            perceptron placed between the centroids
            of X[curr_training_samples].

        centroids : array of shape = [n_classes,n_features]
            updated centroids of each class considering the
            distribution of X[curr_training_samples].

        """

        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = classes.size
        weights = np.zeros((n_classes, n_features), float)
        bias = np.zeros((n_classes), float)

        # Vector indicating the remaining classes in eval_X/eval_y
        curr_classes = np.zeros((n_classes), int)

        mask_incl_samples = np.asarray(curr_training_samples, dtype=bool)

        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples

        midpt_weights = np.ones(n_classes)

        for i in range(0, n_classes):
            # Select instances from a single class
            mask_c = classes[i] == y

            mask_sel = np.logical_and(mask_c, mask_incl_samples)

            c = X[mask_sel]
            # midpt_weights[i] = 1 - np.mean(sample_weights[mask_sel])
            if c.size:
                # Update centroid of class
                if np.sum(sample_weights[mask_sel]) == 0:
                    centroids[i,] = np.average(c, axis=0,
                                               weights=None)
                else:
                    centroids[i,] = np.average(c, axis=0,
                                               weights=sample_weights[mask_sel])
                # Indicate its presence
                curr_classes[i] = 1

        idx_curr_classes = np.where(curr_classes > 0)

        if curr_classes.sum() >= 2:  # More than 2 classes remain
            # Pairwise distance between current classes
            dist_classes = squareform(pdist(centroids[idx_curr_classes[0]]))
            np.fill_diagonal(dist_classes, np.inf)

            # Identify the two farthest away
            closest_dist = np.unravel_index(np.argmin(dist_classes),
                                            dist_classes.shape)

            idx_class_1 = idx_curr_classes[0][closest_dist[0]]
            idx_class_2 = idx_curr_classes[0][closest_dist[1]]

        else:  # Only one class remains
            # Pairwise distance between all classes in the problem
            dist_classes = squareform(pdist(centroids))
            np.fill_diagonal(dist_classes, np.inf)

            # Remaining class
            idx_class_1 = idx_curr_classes[0][0]
            # Most distant from class_1
            idx_class_2 = np.argmin(dist_classes[idx_class_1,])

            # Difference vector between selected classes
        diff_vec = centroids[idx_class_1,] - centroids[idx_class_2,]

        if not np.any(diff_vec):
            # print('Equal classes centroids!')
            w_p = 0.01 * np.ones((n_features), float)
            w_p = w_p / sqrt(((w_p) ** (2)).sum())
        else:
            # Normal vector of diff_vec
            w_p = diff_vec / sqrt(((diff_vec) ** (2)).sum())

        sum_vec = (midpt_weights[idx_class_1] * centroids[idx_class_1,] +
                   midpt_weights[idx_class_2] * centroids[idx_class_2,]) / np.sum(midpt_weights)

        theta_p = np.dot(-w_p, sum_vec)

        # Weights of linear classifier
        weights[idx_class_1,] = w_p
        weights[idx_class_2,] = -w_p

        # Bias of linear classifier
        bias[idx_class_1,] = theta_p
        bias[idx_class_2,] = -theta_p

        assert not np.isnan(theta_p)

        # Generate classifier
        #perc = self.base_estimator_(max_iter=1)
        perc = SGDClassifier(loss="perceptron", eta0=1.e-17, max_iter=1,
                              learning_rate="constant", penalty=None)

        perc.classes_ = classes
        perc.fit(X, y)

        # Set classifier's weigths and bias
        perc.coef_ = weights
        perc.intercept_ = bias

        # Return perceptron
        return perc, centroids

    def _generate_pool(self, X, y, curr_training_samples,
                       sample_weights=None):
        """
        Generates the classifiers in the pool of classifiers
        ("estimators_") using the SGH method.

        In each iteration of the method, a hyperplane is
        placed in the midpoint between the controids of the
        two most distant classes in the training data.
        Then, the newly generated classifier is tested over
        all samples and the ones it correctly labels are
        removed from the set.
        In the following iteration, a new hyperplane is
        created based on the classes of the remaining samples
        in the training set.
        The method stops when no sample remains in the training set.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The training data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        curr_training_samples : array of shape = [n_samples]
            array of ones and zeros ('1','0'), indicating which
            samples in X are to be used for training.
            If all, leave blank.


        Returns
        -------
        self
        """

        # Input data and samples included in the training
        n_samples, n_features = X.shape

        # Labels of the correct classifier for each training sample
        corr_classif_lab = np.zeros((n_samples), int)

        # Pool size
        n_perceptrons = 0

        n_err = 0
        max_err = 3

        # Problem's classes
        classes = np.unique(y)
        n_classes = classes.size

        # Centroids of each class
        centroids = np.zeros((n_classes, n_features), float)

        self.input_data = []

        # While there are still misclassified samples
        while curr_training_samples.sum() > 0 and n_err < max_err:
            # Generate classifier
            self.input_data.append(deepcopy(curr_training_samples))

            curr_perc, centroids = self._build_Perceptron(X, y,
                                                          curr_training_samples,
                                                          centroids, sample_weights)

            # Add classifier to pool
            self.estimators_.append(deepcopy(curr_perc))

            # Obtain set with instances that weren't correctly classified yet
            idx_curr_training_samples = np.where(curr_training_samples > 0)
            eval_X = X[idx_curr_training_samples[0]]
            eval_y = y[idx_curr_training_samples[0]]

            # Evaluate generated classifier over eval_X
            out_curr_perc = self.estimators_[n_perceptrons].predict(eval_X)

            # Identify correctly classified samples
            idx_correct_eval = (out_curr_perc == eval_y).nonzero()

            # Exclude correctly classified samples from current training set
            curr_training_samples[
                idx_curr_training_samples[0][idx_correct_eval[0]]] = 0

            # Set classifier label for the correctly classified instances
            corr_classif_lab[idx_curr_training_samples[0][
                idx_correct_eval[0]]] = n_perceptrons
            # Increase pool size
            n_perceptrons += 1
            n_err += 1

        # Update pool size
        self.n_estimators = n_perceptrons
        # Update classifier labels
        self.correct_classif_label = corr_classif_lab

        return self

"""##PERCEPTON"""

from sklearn.linear_model import Perceptron

import numpy as np
import math
from deslib.util.prob_functions import softmax
import warnings

# from sklearn._stochastic_gradient import BaseSGDClassifier


class PerceptronP(Perceptron):

    def __init__(self, penalty=None, alpha=0.0001, fit_intercept=True,
                 max_iter=1000, tol=1e-3, shuffle=True, verbose=0, eta0=1.0,
                 n_jobs=None, random_state=0, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5,
                 class_weight=None, warm_start=False):
        super().__init__(
            penalty=penalty, alpha=alpha, fit_intercept=fit_intercept,
            max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, eta0=eta0,
            n_jobs=n_jobs, random_state=random_state, early_stopping=early_stopping,
            validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
            class_weight=class_weight, warm_start=warm_start)
        warnings.filterwarnings('ignore')

    def predict_proba(self, X):
        dec_f = self.decision_function(X)
        if len(dec_f.shape) > 1:
            # multi-class: use softmax
            proba = softmax(dec_f)
        else:
            # two-class: use sigmoid

            # suppress warnings
            warnings.filterwarnings('ignore')
            e = np.vectorize(np.exp)
            # limited_scores = 1 / (1 + e(-dec_f))
            limited_scores_normalized = 1 / (1 + e(-dec_f/np.sqrt(np.dot(self.coef_, self.coef_.T))))
            limited_scores_normalized = limited_scores_normalized.ravel()
            proba = np.zeros((np.asarray(X).shape[0], 2))
            for m in np.arange(0, proba.shape[0]):
                proba[m, 1] = limited_scores_normalized[m]
                proba[m, 0] = 1 - limited_scores_normalized[m]

        return proba

"""##OLP"""

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
#from sgh import SGH #usei o arquivo que vc me mandou
from deslib.base import BaseDS
#from src.utils import disagreement, cosine_distance #nao consegui importar, mas achei os métodos
import functools
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.a_priori import APriori
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.mla import MLA
from sklearn.calibration import CalibratedClassifierCV

from copy import deepcopy

#from perceptronp import PerceptronP

from deslib.util.instance_hardness import kdn_score

from sklearn.utils.validation import check_X_y
from sklearn.exceptions import NotFittedError
from collections import Counter


class OLP(BaseDS):
    """
    Online Local Pool (OLP).

    This technique dynamically generates and selects a pool of classifiers
    based on the local region each given
    query sample is located, if such region has any degree of
    class overlap. Otherwise, the technique uses the
    KNN rule for obtaining the query sample's label.

    Parameters
    ----------

    n_classifiers : int (default = 7)
             The size of the pool to be generated for each query instance.

    k : int (Default = 7)
        Number of neighbors used to obtain the Region of Competence (RoC).

    IH_rate : float (default = 0.0)
        Hardness threshold used to identify when to generate the local
        pool or not.

    ds_tech : str (default = 'ola')
        DCS technique to be coupled to the OLP.


    References
    ----------

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin, Online local
    pool generation for
    dynamic classi
er selection, Pattern Recognition 85 (2019) 132-148.

    """

    def __init__(self, n_classifiers=5, k=7, IH_rate=0.0, ds_tech='mcb',
                 knne_roc=False):

        super(OLP, self).__init__(k, IH_rate=IH_rate, n_jobs=1)

        self.name = 'OLP'
        self.pool_classifiers = []
        self.knne_roc = knne_roc
        self.n_jobs = 1

        self.ds_tech = ds_tech
        self.n_classifiers = n_classifiers
        self.k = k
        self.IH_rate = IH_rate
        self._return_report = None
        self.report = None

    def fit(self, X, y):
        """
        Prepare the model by setting the KNN algorithm and
        calculates the information required to apply the OLP

        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features] with the data.

        y : class labels of each sample in X.

        Returns
        -------
        self
        """
        check_X_y(X, y)
        self._set_dsel(X, y)
        self._set_region_of_competence_algorithm()
        self._fit_region_competence(X, y)

        # Set knne
        if self.n_classes > 2 or self.knne_roc is None:
            self.knne_roc = False

        self._return_report = False

        # Calculate the KDN score of the training samples
        if self.IH_rate >= 0:
            self.hardness, _ = kdn_score(X, y, self.k)

        return self

    def _set_region_of_competence_algorithm(self):

        knn_class = functools.partial(KNeighborsClassifier,
                                      n_jobs=self.n_jobs,
                                      algorithm="brute")

        self.knn_class_ = knn_class

        self.roc_algorithm_ = self.knn_class_(n_neighbors=self.k)

    def _set_dsel(self, X, y):
        """
        Get information about the structure of the data
        (e.g., n_classes, N_samples, classes)

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.

        Returns
        -------
        self
        """
        self.DSEL_data = X
        self.DSEL_target = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.n_samples = self.DSEL_target.size

        # There is no pool, so the DSEL can't be preprocessed
        # self.processed_dsel, self.BKS_dsel = self._preprocess_dsel()

        return self

    def _validate_pool(self):
        """Check the n_estimator attribute."""
        if self.n_classifiers < 0:
            raise ValueError("n_classifiers must be greater than zero, "
                             "got {}.".format(self.n_classifiers))

    def _generate_local_pool(self, query):
        """
        Local pool generation.

        This procedure populates the "pool_classifiers" based on
        the query sample's neighborhood.
        Thus, for each query sample, a different pool is created.

        In each iteration, the training samples near the query
        sample are singled out and a
        subpool is generated using the
        Self-Generating Hyperplanes (SGH) method.
        Then, the DCS technique selects the best classifier in the
        generated subpool and it is added to the local pool.
        In the following iteration, the neighborhood is increased
        and another SGH-generated subpool is obtained
        over the new neighborhood, and again the DCS technique
        singles out the best in it, which is then added to the local pool.
        This process is repeated until the pool reaches "n_classifiers".

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample.

        Returns
        -------
        self

        References
        ----------

        M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin,
        On the characterization of the
        oracle for dynamic classi
er selection, in: International
        Joint Conference on Neural Networks,
        IEEE, 2017, pp. 332-339.
        """
        n_samples, _ = self.DSEL_data.shape

        self.pool_classifiers = []
        self.neighborhood = None

        n_err = 0
        max_err = 2 * self.n_classifiers

        curr_k = self.k
        self.classifiers_competence = []

        self.subpools = []

        # Classifier count
        # n = 0

        for n in range(self.n_classifiers): # while n < self.n_classifiers and n_err < max_err:

            subpool = SGH(base_estimator=PerceptronP)

            included_samples = np.zeros(n_samples, int)

            if self.knne_roc:
                idx_neighb = np.array([], dtype=int)

                # Obtain neighbors of each class individually
                for j in np.arange(0, self.n_classes):
                    # Obtain neighbors from the classes in the RoC
                    if np.any(self.classes[j] == self.DSEL_target[
                        self.neighbors[0][np.arange(0,
                                                    curr_k)]]):
                        nc = np.where(self.classes[j] == self.DSEL_target[
                            self.neighbors[0]])
                        idx_nc = self.neighbors[0][nc]
                        idx_nc = idx_nc[
                            np.arange(0, np.minimum(curr_k, len(idx_nc)))]
                        idx_neighb = np.concatenate((idx_neighb, idx_nc),
                                                    axis=0)

            else:
                idx_neighb = np.asarray(self.neighbors)[0][np.arange(0, curr_k)]

            # if it was not defined yet, smallest neighborhood
            if self.neighborhood is None:
                self.neighborhood = deepcopy(idx_neighb)

            # Indicate participating instances in the training of the subpool
            included_samples[idx_neighb] = 1

            curr_classes, count_classes = np.unique(self.DSEL_target[idx_neighb],
                                                    return_counts=True)

            # if self._return_report:
            # If there are +1 classes in the local region
            if len(curr_classes) > 1:

                # Obtain SGH pool
                subpool.fit(self.DSEL_data, self.DSEL_target, included_samples)
                self.subpools.append(deepcopy(subpool))
                if len(subpool.estimators_) > 1:
                    # Adjust chosen DCS technique parameters
                    if self.ds_tech == 'ola':
                        ds = OLA(subpool, k=len(idx_neighb), knne=self.knne_roc)
                    elif self.ds_tech == 'lca':
                        ds = LCA(subpool, k=len(idx_neighb), knne=self.knne_roc)
                    elif self.ds_tech == 'mcb':
                        ds = MCB(subpool, k=len(idx_neighb), knne=self.knne_roc)
                    elif self.ds_tech == 'mla':
                        ds = MLA(subpool, k=len(idx_neighb), knne=self.knne_roc)
                    elif self.ds_tech == 'a_priori':
                        ds = APriori(subpool, k=len(idx_neighb), knne=self.knne_roc)
                    elif self.ds_tech == 'a_posteriori':
                        ds = APosteriori(subpool, k=len(idx_neighb), knne=self.knne_roc)

                    # Fit ds technique
                    ds.fit(self.DSEL_data,
                           self.DSEL_target)

                    # True/False vector of selected neighbors
                    neighb = np.in1d(self.neighbors,
                                     idx_neighb)

                    # Set distances and neighbors of the query sample
                    # (already calculated)
                    ds.distances = np.asarray(
                        [self.distances[0][neighb]])  # Neighborhood
                    ds.neighbors = np.asarray(
                        [self.neighbors[0][neighb]])  # Neighborhood

                    # ds.DFP_mask = np.ones(len(subpool))

                    # Estimate competence
                    comp = ds.estimate_competence(query, np.atleast_2d(idx_neighb),
                                                  np.atleast_2d(ds.distances),
                                                  np.atleast_2d(
                                                      ds._predict_base(query)))

                    # Select best classifier in subpool
                    sel_c = ds.select(comp)
                else:
                    sel_c = [0]

                # Add to local pool
                self.pool_classifiers.append(deepcopy(subpool[sel_c[0]]))

                if self._return_report:
                    # based on the sgh pool
                    self.report['selected-classifier'].append(sel_c[0])  # selected classifiers at each level (id)

                    # based on neighborhood
                    self.report['roc'].append(idx_neighb)
                    self.report['roc-size'].append(len(idx_neighb))  # size of the RoC at each level
                    self.report['kdn'].append(
                        np.mean(self.hardness[idx_neighb]))  # avg loo KDN of the RoC at each level
                    self.report['class-frequency'].append(np.std(count_classes/len(idx_neighb)))
            else:
                self.pool_classifiers.append(curr_classes[0])
                if self._return_report:
                    # based on the sgh pool
                    self.report['selected-classifier'].append(-1)  # selected classifiers at each level (id)
                    # based on neighborhood
                    self.report['roc'].append(idx_neighb)
                    self.report['roc-size'].append(len(idx_neighb))  # size of the RoC at each level
                    self.report['kdn'].append(
                        np.mean(self.hardness[idx_neighb]))  # avg loo KDN of the RoC at each level
                    self.report['class-frequency'].append(0.5)
            # n += 1

            # Increase neighborhood size
            curr_k += 2
            # n_err += 1

        # if it was not defined yet, largest neighborhood
        # if self.neighborhood is None:
        #     self.neighborhood = deepcopy(idx_neighb)

        return self

    def select(self, query):
        """
        Obtains the votes of each classifier given a query sample.

        Parameters
        ----------
        query : array of shape = [n_features] containing the test sample

        Returns
        -------
        votes : array of shape = [len(pool_classifiers)] with the
        class yielded by each classifier in the pool

        """

        votes = np.zeros(len(self.pool_classifiers), dtype=int)
        votes_nn = np.zeros((len(self.neighborhood), len(self.pool_classifiers)))
        for clf_idx, clf in enumerate(self.pool_classifiers):
            try:
                votes[clf_idx] = clf.predict(query)[0]
                votes_nn[:, clf_idx] = clf.predict(self.DSEL_data[self.neighborhood, :])
            except AttributeError:
                votes[clf_idx] = clf
                votes_nn[:, clf_idx] = np.nan
            except TypeError:
                votes[clf_idx] = clf
                votes_nn[:, clf_idx] = np.nan

        if self._return_report:
            # based on the local pool
            self.report['disagreement'].append(disagreement(
                self.DSEL_target[self.neighborhood], votes_nn))  # avg of the dis of the local pool
            self.report['cosine-distance'].append(cosine_distance(self.pool_classifiers))  # avg of the cosine distance of the local pool

        return votes

    def classify_with_ds(self, query):
        """
        Predicts the label of the corresponding query sample.

        The prediction is made by aggregating the votes obtained by
        all selected base classifiers.

        Parameters12t
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """

        # Generate LP
        self._generate_local_pool(query)

        # Predict query label
        if len(self.pool_classifiers) > 0:
            votes = self.select(query)
            counter = Counter(votes)
            # predicted_label = mode(votes)[0]
            predicted_label = counter.most_common(1)[0][0]
        else:
            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]
            counter = Counter(self.DSEL_target[roc])
            predicted_label = counter.most_common(1)[0][0]
            # predicted_label = mode(self.DSEL_target[roc])[0]

        return predicted_label

    def _check_parameters(self):
        """
        Verifies if the input parameters are correct (k)
        raises an error if k < 1.
        """
        if self.k is not None:
            if not isinstance(self.k, int):
                raise TypeError("parameter k should be an integer")
            if self.k <= 1:
                raise ValueError("parameter k must be higher than 1."
                                 "input k is {} ".format(self.k))

        if self.safe_k is not None:
            if not isinstance(self.safe_k, int):
                raise TypeError("parameter safe_k should be an integer")
            if self.safe_k <= 1:
                raise ValueError("parameter safe_k must be higher than 1."
                                 "input safe_k is {} ".format(self.safe_k))

        if not isinstance(self.IH_rate, float):
            raise TypeError(
                "parameter IH_rate should be a float between [0.0, 0.5]")

        self._validate_pool()

    def predict(self, X, return_report=False, X_roc=None):
        """
        Predicts the class label for each sample in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class label for each sample in X.
        """
        # Check if the DS model was trained
        self._check_is_fitted()
        self._return_report = return_report

        if self._return_report:
            self.report = dict()
            self.report['disagreement'] = []  # avg of the dis
            # of the local pool
            self.report['cosine-distance'] = []  # avg of the cosine distance
            # of the local pool

            self.report['selected-classifier'] = []  # selected classifiers at each level (id)
            self.report['roc-size'] = []  # size of the RoC at each level
            self.report['roc'] = []

            self.report['kdn'] = []  # avg loo KDN of the RoC at each level
            self.report['class-frequency'] = []  # sd of class frequencies of the RoC at each level

        n_samples = X.shape[0]
        predicted_labels = np.zeros(n_samples).astype(int)
        for index, instance in enumerate(X):

            instance = instance.reshape(1, -1)

            if X_roc is None:
                # proceeds with DS, calculates the region of competence
                # of the query sample
                self.distances, self.neighbors = self._get_region_competence(
                    instance, k=np.minimum(
                        self.n_samples,
                        (self.n_classes**2) * self.n_classifiers * self.k))
            else:
                self.distances, self.neighbors = self._get_region_competence(
                    X_roc[index].reshape(1, -1), k=np.minimum(
                        self.n_samples,
                        (self.n_classes ** 2) * self.n_classifiers * self.k))

            # predicted_labels[index] = self.classify_with_ds(instance)

            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]

            if self.IH_rate >= 0:
                # If all of its neighbors in the RoC have Instance hardness
                # (IH) below or equal to IH_rate, use KNN
                if np.all(self.hardness[np.asarray(roc)] <= self.IH_rate):
                    y_neighbors = self.DSEL_target[roc]
                    counter = Counter(y_neighbors)
                    predicted_labels[index] = counter.most_common(1)[0][0]

                    if self._return_report:
                        # based on the sgh pool
                        self.report['selected-classifier'].extend([-1 for j in range(self.n_classifiers)])  # selected classifiers at each level (id)
                        # based on neighborhood
                        self.report['roc'].extend([0 for j in range(self.n_classifiers)])
                        self.report['roc-size'].extend([0 for j in range(self.n_classifiers)])   # size of the RoC at each level
                        self.report['kdn'].extend([0 for j in range(self.n_classifiers)])  # avg loo KDN of the RoC at each level
                        self.report['class-frequency'].append([0.5 for j in range(self.n_classifiers)])
                        self.report['disagreement'].append(np.nan)  # avg of the dis of the local pool
                        self.report['cosine-distance'].append(np.nan)  # avg of the cosine distance of the local pool

                    # predicted_labels[index], _ = mode(y_neighbors)

                # Otherwise, generate the local pool for the query instance
                # and use DS for classification
                else:
                    predicted_labels[index] = self.classify_with_ds(instance)
            else:
                predicted_labels[index] = self.classify_with_ds(instance)

            self.neighbors = None
            self.distances = None

        self._return_report = False
        if return_report:
            return predicted_labels, self.report
        else:
            return predicted_labels  # , None

    def classify_with_ds_proba(self, query):
        """
        Predicts the label of the corresponding query sample.

        The prediction is made by aggregating the votes obtained by
        all selected base classifiers.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """

        # Generate LP
        self._generate_local_pool(query)

        # self.classifiers_competence = \
        #     np.asarray(self.classifiers_competence).ravel()

        predicted_class_prob = np.zeros(len(self.classes))

        if True:
            # Predict query label
            if len(self.pool_classifiers) > 0:
                proba = np.zeros((len(self.pool_classifiers), self.n_classes))
                for clf_idx, clf in enumerate(self.pool_classifiers):
                    try:
                        proba[clf_idx, :] = clf.predict_proba(query)[0]
                    except AttributeError:
                        proba[clf_idx, int(clf)] = 1
                predicted_class_prob = np.mean(proba, axis=0)
            else:
                nn = np.arange(0, self.k)
                roc = self.neighbors[0][nn]
                counter = Counter(self.DSEL_target[roc])
                predicted_label = counter.most_common(1)[0][0]
                predicted_class_prob[predicted_label] = 1
                # predicted_label = mode(self.DSEL_target[roc])[0]

        else:
            # Predict query label
            if len(self.pool_classifiers) > 0:
                proba = np.zeros((len(self.pool_classifiers), self.n_classes))
                for clf_idx, clf in enumerate(self.pool_classifiers):
                    clf.coef_ = np.asarray([clf.coef_[0]])
                    clf.intercept_ = np.asarray([clf.intercept_[0]])
                    clf_calibrated = CalibratedClassifierCV(
                        base_estimator=clf, method='sigmoid',
                        cv='prefit').fit(
                        self.DSEL_data[self.neighborhood, :],
                        self.DSEL_target[self.neighborhood])
                    # clf_calibrated = CalibratedClassifierCV(
                    #     base_estimator=clf, method='sigmoid', cv='prefit').fit(
                    #     self.DSEL_data[
                    #     self.neighbors[0][np.arange(0, 3 * self.k)], :],
                    #     self.DSEL_target[
                    #         self.neighbors[0][np.arange(0, 3 * self.k)]])
                    proba[clf_idx, :] = clf_calibrated.predict_proba(query)[0]

                    predicted_class_prob = np.mean(proba, axis=0)
            else:
                nn = np.arange(0, self.k)
                roc = self.neighbors[0][nn]
                y_neighbors = self.DSEL_target[roc]

                for c in np.arange(0, len(self.classes)):
                    predicted_class_prob[c] = \
                        np.sum(y_neighbors == self.classes[c]) / len(roc)

        return predicted_class_prob

    def predict_proba(self, X, return_report=False):

        # Check if the DS model was trained
        self._check_is_fitted()
        self._return_report = return_report

        n_samples = X.shape[0]
        predicted_class_prob = np.zeros((n_samples, self.n_classes)).astype(
            float)
        for index, instance in enumerate(X):

            instance = instance.reshape(1, -1)

            #proceeds with DS, calculates the region of
            #competence of the query sample
            self.distances, self.neighbors = self._get_region_competence(
                instance, k=np.minimum(
                    self.n_samples,
                    (self.n_classes**2) * self.n_classifiers * self.k))

            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]

            # If all of its neighbors in the RoC have Instance hardness (IH)
            # below or equal to IH_rate, use KNN
            if np.all(self.hardness[np.asarray(roc)] <= self.IH_rate):
                y_neighbors = self.DSEL_target[roc]
                for c in np.arange(0, len(self.classes)):
                    predicted_class_prob[index, c] = \
                        np.sum(y_neighbors == self.classes[c]) / len(roc)

            # Otherwise, generate the local pool for the query instance and
            # use DS for classification
            else:
                predicted_class_prob[index, :] = self.classify_with_ds_proba(
                    instance)

            self.neighbors = None
            self.distances = None

        return predicted_class_prob

    def _check_is_fitted(self):
        """ Verify if the dynamic selection algorithm was fitted. Raises
        an error if it is not fitted.

        Raises
        -------
        NotFittedError
            If the DS method is was not fitted, i.e., `self.roc_algorithm`
             or `self.processed_dsel`
             were not pre-processed.

        """
        if self.roc_algorithm_ is None:
            raise NotFittedError("DS method not fitted, "
                                 "call `fit` before exploiting the model.")
