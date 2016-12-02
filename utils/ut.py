#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict


def get_rest(key, dictionary):
    return [dictionary[k] for k in dictionary if k != key]


def tfidf(X):
    # Jako argument przyjmuje tablicę dokumentów
    pass


def parallelSort(X, Y, classes):
    sortedDocs = dict((c, []) for c in classes)
    for x, y in zip(X, Y):
        sortedDocs[y].append(x)
    return sortedDocs


def mergeDicts(X):
    merged = defaultdict(int)
    for d in X:
        for k, v in d.items():
            merged[k] += v
    return merged


def allTermsMerge(X, terms):
    merged = dict((t, 0) for t in terms)
    for k, v in X.items():
        merged[k] += v
    return merged


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def accuracy(KlasaA, KlasaB, bayesModel):
    klasaATrue = []
    klasaBTrue = []
    for i in range(0, len(KlasaA)):
        # Klasa A
        if bayesModel.classify(KlasaA[i]) == 'Klasa A':
            klasaATrue.append(1)
        else:
            klasaATrue.append(0)
        # Klasa B
        if bayesModel.classify(KlasaB[i]) == 'Klasa B':
            klasaBTrue.append(1)
        else:
            klasaBTrue.append(0)
    accA = klasaATrue.count(1) / len(klasaATrue)
    accB = klasaATrue.count(1) / len(klasaBTrue)
    mean = (accA + accB) / 2

    print "========Wyniki========"
    print "Dokładność klasyfikacji klasy A: " + str(accA)
    print "Dokładność klasyfikacji klasy B: " + str(accB)
    print "Średnia dokładność: " + str(mean)

