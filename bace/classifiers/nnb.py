# -*- coding: utf-8 -*-
# Author: Krzysztof Joachimiak

import numpy as np
from sklearn.preprocessing import LabelBinarizer
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
        features_doc_logprob = self.safe_matmult(X, features_weights.T)
        return (features_doc_logprob * - np.exp(1)) + self.class_log_proba_

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

        self._class_log_prob()
        self._update_complement_features(X, y_one_hot)
        self.is_fitted = True

    def _class_log_prob(self):
        '''
        Compute complement probability of class occurence
        '''
        all_samples_count = np.float64(np.sum(self.class_count_))
        self.complement_class_counts_ = self.class_count_.dot(get_complement_matrix(len(self.class_count_)))
        self.complement_class_proba_ = (self.complement_class_count_  / all_samples_count) ** -1
        #self.class_log_proba_ = np.log(self.complement_class_counts_)


    def _reset(self):
        '''

        Reset object params for refit

        '''
        self.classes_ = None
        self.class_counts_ = None
        self.complement_features_ = None
        self.complement_class_counts_ = None