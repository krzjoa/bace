# -*- coding: utf-8 -*-
# Author: Krzysztof Joachimiak 2016

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from bace.base import BaseNB
from bace.utils import inherit_docstring, safe_matmult


@inherit_docstring
class ComplementNB(BaseNB):

    '''
    Complement Naive Bayes classifier

    Parameters
    ----------
    alpha: float
        Smoothing parameter
    weight_normalized: bool, default False
        Enable Weight-normalized Complement Naive Bayes method.

    Attributes
    ----------
    alpha_sum_ : int
        Sum of alpha params
    classes_ : array, shape (n_classes,)
        Classes list
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.

    Examples
    --------
    >>> from sklearn.datasets import fetch_20newsgroups
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from bace import ComplementNB
    Prepare data
    >>> vectorizer = CountVectorizer()
    >>> categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
    Train set
    >>> newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
    >>> train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    Test set
    >>> newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
    >>> test_vectors = vectorizer.transform(newsgroups_test.data)
    >>> clf = ComplementNB()
    >>> clf.fit(newsgroups_train, train_vectors).accuracy_score(newsgroups_test, test_vectors)

    References
    ----------
    Rennie J. D. M., Shih L., Teevan J., Karger D. R.  (2003).
    Tackling the Poor Assumptions of Naive Bayes Text Classifiers

    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
    '''

    def __init__(self, alpha=1.0, weight_normalized=False):
        super(ComplementNB, self).__init__()

        # Params
        self.alpha = alpha
        self._check_alpha_param()

        if weight_normalized:
            self._not_implemented_yet('Weighted Complement Naive Bayes is not implemented yet!')

        self.weight_normalized = weight_normalized

        # Computed attributes
        self.complement_features_ = None
        self.alpha_sum_ = None

    def fit(self, X, y):
        self._reset()
        self._partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        self._partial_fit(X, y, classes=classes, first_partial_fit=not self.is_fitted)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def predict_log_proba(self, X):
        self._check_is_fitted()
        denominator = np.sum(self.complement_features, axis=0) + self.alpha_sum_
        features_weights = np.log((self.complement_features + self.alpha) / denominator)

        if self.weight_normalized:
            features_weights = features_weights / np.sum(np.absolute(features_weights), axis=0)
            # from scipy.misc import logsumexp
            # features_weights = features_weights - logsumexp(features_weights, axis=0)

        features_doc_logprob = safe_matmult(X, features_weights.T)

        return (features_doc_logprob * - np.exp(-1)) + self.class_log_proba_


    # Fitting model

    def _partial_fit(self, X, y, classes=None, first_partial_fit=None):

        if first_partial_fit and not classes:
            raise ValueError("classes must be passed on the first call "
                         "to partial_fit.")

        if not self.is_fitted:
            self.alpha_sum_ = X.shape[1] * self.alpha

        if classes:
            self.classes_ = classes

        lb = LabelBinarizer()
        y_one_hot = lb.fit_transform(y)
        self.class_count_ = np.sum(y_one_hot, axis=0)

        if not self.classes_:
            self.classes_ = lb.classes_

        #self._class_log_prob()
        self._update_complement_features(X, y_one_hot)
        self.is_fitted = True

        #print "CNB class count", self.class_count_
        #print "CNB complement class count", self.complement_class_count_
        #print "CNB features", self.features_
        #print "CNB complement features", self.complement_features

    def _reset(self):
        '''

        Reset object params for refit

        '''
        self.classes_ = None
        self.class_counts_ = None
        self.complement_features_ = None
        self.complement_class_counts_ = None
