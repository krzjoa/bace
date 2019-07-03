# -*- coding: utf-8 -*-
# Author: Krzysztof Joachimiak 2016

import numpy as np
from bace.base import BaseNB
from bace.utils import inherit_docstring

# TODO: check weight normalization


@inherit_docstring
class ComplementNB(BaseNB):

    '''
    Complement Naive Bayes classifier

    References
    ----------
    Rennie J. D. M., Shih L., Teevan J., Karger D. R.  (2003).
    Tackling the Poor Assumptions of Naive Bayes Text Classifiers

    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

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
    '''

    def __init__(self, alpha=1.0, weight_normalized=False):
        super(ComplementNB, self).__init__()

        # Params
        self.alpha = alpha
        self._check_alpha_param()
        self.weight_normalized = weight_normalized

        # Computed attributes
        self.complement_features_ = None
        self.alpha_sum_ = None

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def predict_log_proba(self, X):
        self._check_is_fitted()
        denominator = np.sum(self.complement_features, axis=0) + self.alpha_sum_
        features_weights = np.log((self.complement_features + self.alpha) / denominator)

        if self.weight_normalized:
            features_weights /= np.abs(features_weights).sum(axis=1, keepdims=True)

        features_doc_logprob = X @ features_weights
        return self.class_log_proba_ - features_doc_logprob
        #return (features_doc_logprob * - np.exp(-1)) + self.class_log_proba_

    # Fitting model
    def _partial_fit(self, X, y, classes=None, first_partial_fit=None):
        X, y_one_hot = self._prepare_X_y(X, y, first_partial_fit, classes)
        #self._class_log_prob()
        self._update_complement_features(X, y_one_hot)
        self.is_fitted = True