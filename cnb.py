#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import BaseDiscreteNB, MultinomialNB
from sklearn.utils import check_X_y
import numpy as np
from collections import Counter
import bayes_utils as nb

# Author; Krzysztof Joachimiak


class ComplementNB(object):


    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.counts = None

    def fit(self, X, y, sample_weight=None):
        pass

    # def _count(self, X, Y):
    #     """Count and smooth feature occurrences."""
    #     if np.any((X.data if issparse(X) else X) < 0):
    #         raise ValueError("Input X must be non-negative")
    #     self.feature_count_ += safe_sparse_dot(Y.T, X)
    #     self.class_count_ += Y.sum(axis=0)



    def partial_fit(self, X, y):
        check_X_y(X,y)









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

            #



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
					
				
