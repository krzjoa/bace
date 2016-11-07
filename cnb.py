#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.utils import check_X_y
import numpy as np
from collections import Counter
from bayes_utils import get_rest
import operator
from sklearn.naive_bayes import MultinomialNB, BaseNB

# Author; Krzysztof Joachimiak



class ComplementNB(object):


    def __init__(self, alpha=1.0):
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

    def predict_proba(self):
        pass

    def predict_log_proba(self):
        pass

    def score(self, X, y):
        pass


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








        # def __init__(self,X,Y, alfa=1):
        # 	""" X and Y must be numpy arrays """
        # 	self.X = X
        # 	self.Y = Y
        # 	self.alfa=alfa #Smoothing parameter
        # 	self.classes = np.unique(self.Y)
        # 	self.classFrequencies = dict(Counter(self.Y))
        # 	self.setSize = len(self.X)
        # 	self.classProbability = self._get_probability_of_class()
        # 	#Liczebność termów w poszczególnych klasach
        # 	self.complementTermsFrequencies = self._get_complement_frequencies()
        # 	self.classCounts = self._get_counts_in_classes()
        # 	self.complementTermsProbabilities = self._get_complement_probabilities()


        # def classify(self,X):
        # 	probabs = self.probabilities(X)
        # 	return max(probabs, key=probabs.get)



        # def predict(self, X):
        #     predictions = []
        #     for x in X:
        #         predictions.append(self.classify(x))
        #     return predictions
    # def _get_counts(self):
	# 	v = np.zeros(len(self.X[0]))
	# 	for i in self.X: v+=i
	# 	return v
	#
	# def _get_complement_frequencies(self):
	# 	cmpfreq = []
	# 	for c in self.classes:
	# 		v = np.zeros(len(self.X[0]))
	# 		for i, j  in zip(self.X, self.Y):
	# 			if j != c: v+=i
	# 		cmpfreq.append(v)
	# 	return cmpfreq
	#
	# def _get_complement_probabilities(self):
	# 	cmpprob = []
	# 	for vector, count in zip(self.complementTermsFrequencies, self.classCounts):
	# 		cmpprob.append(vector.astype(float)/count)
	# 	return cmpprob
	#
	# def _get_counts_in_classes(self):
	# 	counts = []
	# 	for i in self.complementTermsFrequencies: counts.append(sum(i))
	# 	return counts
	#
	# def _get_probability_of_class(self):
	# 	return dict((c,float(self.classFrequencies[c])/float(self.setSize)) for c in self.classes)
	#
	# def probabilities(self,X):
	# 	probabilities = dict()
	# 	for j, c in zip(self.complementTermsProbabilities, self.classes):
	# 		denominator = 1
	# 		for power, base in zip(X,j):
	# 			if power !=0:
	# 				result = pow(base, power)
	# 				denominator*= result if result!=0 else 1
	# 		if denominator==0: print denominator
	# 		fraction = self.classProbability[c]*(1/denominator)
	# 		probabilities[c] = fraction
	# 	return probabilities
	#
					
				
