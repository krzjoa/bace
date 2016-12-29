#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix

def get_complement_matrix(size):
    ones = np.ones((size,))
    return np.logical_not(np.diag(ones))


def safe_mult(input_array, internal_array):
    if isinstance(input_array, csr_matrix):
        input_array = input_array.toarray()
    return input_array * internal_array


def safe_matmult(input_array, internal_array):
    if isinstance(input_array, csr_matrix):
        input_array = input_array.toarray()
    return input_array.dot(internal_array.T)

# def binarize_labels(y_input, classes_vector=None, classes_counts=None):
#
#     if (classes_vector and not classes_counts) or (not classes_vector and classes_counts):
#         raise Exception('You should pass both classes vector and classes counts!')
#
#     classes_vector = classes_vector
#     classes_counts = classes_counts
#
#     y_unique, y_counts = np.unique(y_input, return_counts=True)
#
#
#
#
# def update_bin_labels(y_input, classes_vector, classes_counts):
#     y_unique, y_counts = np.unique(y_input, return_counts=True)
#
#     # Check if y_unique has items,
#     # which don't occur in the classes_vector
#
#     classes_diff = np.setdiff1d(y_unique, classes_vector)
#
#     if classes_diff:
#         classes_vector = np.hstack((classes_vector, classes_diff))
#


#
# def get_rest(key, dictionary):
#     return [dictionary[k] for k in dictionary if k != key]




# def parallelSort(X, Y, classes):
#     sortedDocs = dict((c, []) for c in classes)
#     for x, y in zip(X, Y):
#         sortedDocs[y].append(x)
#     return sortedDocs
#
#
# def mergeDicts(X):
#     merged = defaultdict(int)
#     for d in X:
#         for k, v in d.items():
#             merged[k] += v
#     return merged
#
#
# def allTermsMerge(X, terms):
#     merged = dict((t, 0) for t in terms)
#     for k, v in X.items():
#         merged[k] += v
#     return merged
