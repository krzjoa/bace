#!/usr/bin/env python
# -*- coding: utf-8 -*-

import operator
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from utils.ut import get_rest
from utils.bayes_utils import get_complement_matrix
from base import BaseNB
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix

# Author: Krzysztof Joachimiak


class ComplementNB(BaseNB):

    '''
    Complement Naive Bayes classifier

    Parameters
    ----------
    alpha: float
        Smoothing parameter
    weighted: bool, default False
        Enable Weight-normalized Complement Naive Bayes method.

    References
    ----------
    Rennie J. D. M., Shih L., Teevan J., Karger D. R.  (2003).
    Tackling the Poor Assumptions of Naive Bayes Text Classifiers

    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
    '''

    def __init__(self, alpha=1.0, weighted=False):
        super(ComplementNB, self).__init__()

        # Params
        self.alpha = alpha
        self.alpha_sum_ = None
        self._check_alpha_param()

        if weighted:
            self._not_implemented_yet('Weighted Complement Naive Bayes is not implemented yet!')

        # TODO: Implement Weighted Naive Bayes
        self.weighted = weighted

        # Computed attributes
        self.classes_ = None
        self.class_counts_ = None
        self.complement_class_log_proba_ = None
        self.complement_features_ = None
        self.complement_class_counts_ = None

    def fit(self, X, y):
        self._reset()
        self._partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        self._partial_fit(X, y, classes=classes, first_partial_fit=not self.is_fitted)
        return self

    def predict_proba(self, X):
        # TODO: Handle float exponent error
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def predict_log_proba(self, X):
        self._check_is_fitted()
        denominator = np.sum(self.complement_features, axis=0) + self.alpha_sum_
        features_logprob = np.log((self.complement_features + self.alpha) / denominator)
        features_doc_logprob = self.safe_matmult(X, features_logprob.T)
        return (features_doc_logprob * -1) + self.complement_class_log_proba_

    def get_params(self):
        return self.__dict__

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

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
        self.class_counts_ = np.sum(y_one_hot, axis=0)

        if not self.classes_:
            self.classes_ = lb.classes_

        self._complement_class_log_prob()
        self._features_in_class(X, y, y_one_hot)
        self.is_fitted = True

    def _complement_class_log_prob(self):
        all_samples_count = np.float64(np.sum(self.class_counts_))
        self.complement_class_counts_ = self.class_counts_.dot(get_complement_matrix(len(self.class_counts_)))
        self.complement_class_log_proba_ = np.log(self.complement_class_counts_ / all_samples_count)

    def _features_in_class(self, X, y, y_one_hot):
        if not self.is_fitted:
            self.complement_features = X.T.dot(np.logical_not(y_one_hot))
        else:
            self.complement_features += X.T.dot(np.logical_not(y_one_hot))

    def _reset(self):
        self.classes_ = None
        self.class_counts_ = None
        self.complement_class_log_proba_ = None
        self.complement_features_ = None
        self.complement_class_counts_ = None
        self.complement_class_log_proba_ = None