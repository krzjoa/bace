#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
import NBUtils as nb
from operator import add, mul, itemgetter
from prettytable import PrettyTable


class NNB:
	def __init__(self,X,Y, alfa=1):
		""" X and Y must be numpy arrays """
		self.X = X
		self.Y = Y
		self.alfa=alfa #Smoothing parameter
		self.classes = np.unique(self.Y)
		self.classFrequencies = dict(Counter(self.Y))
		self.setSize = len(self.X)
		self.classProbability = self._get_probability_of_class()
		#Liczebność termów w poszczególnych klasach
		self.complementTermsFrequencies = self._get_complement_frequencies()
		self.classCounts = self._get_counts_in_classes()
		self.complementTermsProbabilities = self._get_complement_probabilities()

	def classify(self,X):
		probabs = self.probabilities(X)
		return max(probabs, key=probabs.get)
		
	def predict(self, X):
		predictions = []
		for x in X:
			predictions.append(self.classify(x))
		return predictions	
			
		
	def _get_counts(self):
		v = np.zeros(len(self.X[0]))
		for i in self.X: v+=i
		return v
	
	def _get_complement_frequencies(self):
		cmpfreq = []
		for c in self.classes:
			v = np.zeros(len(self.X[0]))
			for i, j  in zip(self.X, self.Y): 
				if j != c: v+=i
			cmpfreq.append(v)	
		return cmpfreq		
		
	def _get_complement_probabilities(self):
		cmpprob = []
		for vector, count in zip(self.complementTermsFrequencies, self.classCounts):
			cmpprob.append(vector.astype(float)/count)
		return cmpprob		
		
	def _get_counts_in_classes(self):
		counts = []
		for i in self.complementTermsFrequencies: counts.append(sum(i))
		return counts				
		
	def _get_probability_of_class(self):
		return dict((c,float(self.classFrequencies[c])/float(self.setSize)) for c in self.classes)		
	
	def probabilities(self,X):
		probabilities = dict()
		for j, c in zip(self.complementTermsProbabilities, self.classes):
			denominator = 1
			for power, base in zip(X,j):
				if power !=0:
					result = pow(base, power)
					denominator*= result if result!=0 else 1
			fraction = (1/(1-self.classProbability[c]))*(1/denominator)
			probabilities[c] = fraction	
		return probabilities	
			
					
				
