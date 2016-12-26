#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict


def get_rest(key, dictionary):
    return [dictionary[k] for k in dictionary if k != key]


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

