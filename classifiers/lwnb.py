#!/usr/bin/env python
# -*- coding: utf-8 -*-

import operator
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from utils.ut import get_rest


# Author: Krzysztof Joachimiak


class LocallyWeightedNB(object):



    def __init__(self, alpha=1.0):

        raise Exception("Not implemented yet!")

        self.alpha = alpha
        self.counts = None
        self.class_occurences = dict()
        self.complement_counts = None
        self.complement_class_log_probs = dict()
        self.tokens_in_classes = {}
        self.complement_tokens_in_classes = {}
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = False
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y):
        self.class_occurences.update(Counter(y))
        self._comlement_class_log_prob()
        self._tokens_in_class(X, y)
        self._complement_tokens_in_class()
        return self

    def predict(self, X):
        predictions = []
        for row in X:
            class_proba = []
            for class_name in self.complement_class_log_probs:
                class_proba.append((class_name, self._compute_probability(class_name, row)))
            class_proba = sorted(class_proba, key=operator.itemgetter(1), reverse=True)
            predictions.append(class_proba[0][0])
        return predictions

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        return [np.exp(lp) for lp in log_proba]

    def predict_log_proba(self, X):
        predictions = []
        for row in X:
            class_proba = []
            for class_name in self.complement_class_log_probs:
                class_proba.append(self._compute_probability(class_name, row))
            predictions.append(class_proba)
        return predictions

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self):
        return self.__dict__

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

    def _comlement_class_log_prob(self):
        all_samples_count = sum(self.class_occurences.values())
        for key in self.class_occurences:
            self.complement_class_log_probs[key] = np.log(sum(get_rest(key, self.class_occurences))
                                                          / float(all_samples_count))

    def _tokens_in_class(self, X, y):
        if not self.is_fitted:
            self._tokens_in_class_first(X, y)
        else:
            self._tokens_in_class_partial(X, y)

    def _tokens_in_class_first(self, X, y):
        for cl in self.class_occurences.keys():
            mask = np.where(y == cl)[0]
            self.tokens_in_classes[cl] = np.sum(X[mask], axis=0)

    def _tokens_in_class_partial(self, X, y):
        for cl in self.class_occurences.keys():
            mask = np.where(y == cl)[0]
            self.tokens_in_classes[cl] += np.sum(X[mask], axis=0)


    def _complement_tokens_in_class(self):
        for class_name in self.tokens_in_classes:
            self.complement_tokens_in_classes[class_name] = np.sum(get_rest(class_name, self.tokens_in_classes), axis=0)

    def _compute_probability(self, class_name, x_row):
        ctc = self.complement_tokens_in_classes[class_name]
        denominator = sum(ctc) + self.alpha
        return self.complement_class_log_probs[class_name] - (np.sum(x_row * np.log(ctc + self.alpha / denominator)))