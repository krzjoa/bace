# -*- coding: utf-8 -*-
# Author: Krzysztof Joachimiak

import numpy as np
from bace.base import BaseNB
from bace.utils import get_complement_matrix, inherit_docstring


@inherit_docstring
class NegationNB(BaseNB):

    '''
    Negation Naive Bayes classifier

    Parameters
    ----------
    alpha: float
        Smoothing parameter

    References
    ----------
    Komiya K., Sato N., Fujimoto K., Kotani Y. (2011).
    Negation Naive Bayes for Categorization of Product Pages on the Web

    http://www.aclweb.org/anthology/R11-1083.pdf
    '''

    def __init__(self, alpha=1.0):
        super(NegationNB, self).__init__()

        # Params
        self.alpha = alpha
        self.alpha_sum_ = None
        self._check_alpha_param()

        # Computed attributes
        self.classes_ = None
        self.class_counts_ = None

    def predict(self, X):
        return self.classes_[np.argmin(self.predict_log_proba(X), axis=1)]

    def predict_log_proba(self, X):
        self._check_is_fitted()
        denominator = np.sum(self.complement_features, axis=0) + self.alpha_sum_
        features_weights = np.log((self.complement_features + self.alpha) / denominator)
        features_doc_logprob = X @ features_weights
        return self.class_log_proba_ + features_doc_logprob

    def _partial_fit(self, X, y, classes=None, first_partial_fit=None):
        X, y_one_hot = self._prepare_X_y(X, y, first_partial_fit, classes)
        self._class_log_prob()
        self._update_complement_features(X, y_one_hot)
        self.is_fitted = True

    def _class_log_prob(self):
        '''
        Compute complement probability of class occurence
        '''
        all_samples_count = np.float64(np.sum(self.class_count_))
        self.complement_class_counts_ = self.class_count_.dot(get_complement_matrix(len(self.class_count_)))
        self.complement_class_proba_ = (self.complement_class_count_ / all_samples_count) ** -1