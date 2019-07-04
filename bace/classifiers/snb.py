#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from bace.base import BaseNB
from bace.utils import inherit_docstring
from scipy.special import logsumexp

# Author: Krzysztof Joachimiak

@inherit_docstring
class SelectiveNB(BaseNB):

    '''
    Selective Naive Bayes classifier

    Parameters
    ----------
    alpha: float
        Smoothing parameter

    References
    ----------
    Komiya K., Ito Y., Kotani Y. (2013).
    New Naive Bayes Methods using Data from All Classes

    https://github.com/krzjoa/bace/blob/master/papers/snb.pdf
    '''

    _threshold = np.log(0.5)

    def __init__(self, alpha=1.0):
        super(SelectiveNB, self).__init__()

        # Params
        self.alpha = alpha
        self.alpha_sum_ = None
        self._check_alpha_param()

        # Computed attributes
        self.classes_ = None
        self.class_counts_ = None
        self.complement_features_ = None
        self.features_ = None

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def predict_log_proba(self, X):
        self._check_is_fitted()
        return self._predict_log_proba(X)

    # Making predictions
    def _predict_log_proba(self, X):
        '''

        Predict log_proba basing on class prior probability.
        If it exceeds or equals 0.5 threshold, the log_proba is
        computed according to _geq method. Otherwise, the _less
        method is applied.

        Parameters
        ----------
        X: array-like (n_samples, n_features)
            Array of unseen samples

        Returns
        -------
        log_proba: array-like (n_samples, n_classes)
            Log probability matrix

        '''
        _geq_mask = self.class_log_proba_ >= SelectiveNB._threshold
        _less_mask = self.class_log_proba_ < SelectiveNB._threshold
        return _geq_mask * self._geq(X) + _less_mask * self._less(X)

    def _geq(self, X):
        numerator = self._log_proba(X)
        denominator = logsumexp(numerator, axis=1)
        denominator = denominator.reshape(len(denominator), 1)
        return numerator - denominator

    def _less(self, X):
        numerator = self._log_proba(X) + np.log(len(self.classes_) - 1)
        denominator = logsumexp(numerator, axis=1)
        denominator = denominator.reshape(len(denominator), 1)
        return  (numerator - denominator) + np.exp(-1) + np.exp(1)

    def _log_proba(self, X):
        denominator = np.sum(self.features_, axis=0) + self.alpha_sum_
        features_weights = np.log((self.features_ + self.alpha) / denominator)
        features_doc_logprob = X @ features_weights
        return (features_doc_logprob) + self.class_log_proba_

    def _complement_log_proba(self, X):
        denominator = np.sum(self.complement_features_, axis=0) + self.alpha_sum_
        features_weights = np.log((self.complement_features_ + self.alpha) / denominator)
        features_doc_logprob = X @ features_weights
        return (features_doc_logprob) + self.complement_class_log_proba_

    def _partial_fit(self, X, y, classes=None, first_partial_fit=None):
        X, y_one_hot = self._prepare_X_y(X, y, first_partial_fit, classes)
        self._update_complement_features(X, y_one_hot)
        self._update_features(X, y_one_hot)
        self.is_fitted = True