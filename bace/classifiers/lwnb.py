#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from bace.base import BaseNB


# Author: Krzysztof Joachimiak


class LocallyWeightedNB(BaseNB):

    '''
    Locally Weighted Naive Bayes classifier

    Parameters
    ----------
    alpha: float
        Smoothing parameter
    weight_normalized: bool, default False
        Enable Weight-normalized Complement Naive Bayes method.

    References
    ----------
    Frank E., Hall M., Pfahringer B. (2003).

    http://www.cs.waikato.ac.nz/~eibe/pubs/UAI_200.pdf
    '''

    def __init__(self, alpha=1.0, weight_normalized=False):
        super(LocallyWeightedNB, self).__init__()

        # Params
        self.alpha = alpha
        self.alpha_sum_ = None
        self._check_alpha_param()

        if weight_normalized:
            self._not_implemented_yet('Weighted Complement Naive Bayes is not implemented yet!')

        self.weight_normalized = weight_normalized

        # Computed attributes
        self.classes_ = None
        self.class_counts_ = None
        # self.complement_class_log_proba_ = None
        self.class_log_proba_ = None
        self.complement_features_ = None
        # self.complement_class_counts_ = None

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

    def predict_log_proba(self, X):
        self._check_is_fitted()
        denominator = np.sum(self.complement_features, axis=0) + self.alpha_sum_
        features_weights = np.log((self.complement_features + self.alpha) / denominator)

        if self.weight_normalized:
            features_weights = features_weights / np.sum(np.absolute(features_weights), axis=0)

        features_doc_logprob = self.safe_matmult(X, features_weights.T)
        return (features_doc_logprob * -1) + self.class_log_proba_

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

        self._class_log_prob()
        self._features_in_class(X, y_one_hot)
        self.is_fitted = True

    def _class_log_prob(self):
        '''
        Compute complement probability of class occurence
        '''
        all_samples_count = np.float64(np.sum(self.class_counts_))
        self.class_log_proba_ = np.log(self.class_counts_ / all_samples_count)

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
            self.complement_features = X.T.dot(np.logical_not(y_one_hot))
        else:
            self.complement_features += X.T.dot(np.logical_not(y_one_hot))

    def _reset(self):
        '''

        Reset object params for refit

        '''
        self.classes_ = None
        self.class_counts_ = None
        self.class_log_proba_ = None
        self.complement_features_ = None
        self.complement_class_counts_ = None
        self.class_log_proba_ = None