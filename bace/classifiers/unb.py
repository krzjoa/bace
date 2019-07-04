#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from bace.base import BaseNB
from bace.utils import inherit_docstring

# Author: Krzysztof Joachimiak

@inherit_docstring
class UniversalSetNB(BaseNB):

    '''
    Universal-set Naive Bayes classifier

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

    def __init__(self, alpha=1.0):
        super(UniversalSetNB, self).__init__()

        # Params
        self.alpha = alpha
        self.alpha_sum_ = None
        self._check_alpha_param()

        # Computed attributes
        self.classes_ = None
        self.class_counts_ = None
        self.complement_features_ = None
        self.features_ = None

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
        return self._log_proba(X) - self._complement_log_proba(X)

    # Making predictions
    def _complement_log_proba(self, X):
        denominator = np.sum(self.complement_features_, axis=0) + self.alpha_sum_
        features_weights = np.log((self.complement_features_ + self.alpha) / denominator)
        features_doc_logprob = X @ features_weights
        return (features_doc_logprob) + self.complement_class_log_proba_

    def _log_proba(self, X):
        denominator = np.sum(self.features_, axis=0) + self.alpha_sum_
        features_weights = np.log((self.features_ + self.alpha) / denominator)
        features_doc_logprob = X @ features_weights
        return (features_doc_logprob) + self.class_log_proba_

    # Fitting model
    def _partial_fit(self, X, y, classes=None, first_partial_fit=None):
        X, y_one_hot = self._prepare_X_y(X, y, first_partial_fit, classes)
        self._features_in_class(X, y_one_hot)
        self.is_fitted = True

    def _features_in_class(self, X, y_one_hot):
        '''

        Compute complement features counts

        Parameters
        ----------
        X: numpy array (n_samples, n_features)
            Matrix of input samples
        y_one_hot: numpy array (n_samples, n_classes)
            Binary matrix encoding input
        '''
        if not self.is_fitted:
            self.complement_features_ = X.T @ np.logical_not(y_one_hot)
            self.features_ = X.T @ y_one_hot
        else:
            self.complement_features_ += X.T @ np.logical_not(y_one_hot)
            self.features_ += X.T @ y_one_hot
