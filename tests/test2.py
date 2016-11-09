from collections import Counter

import numpy as np

from classifiers.cnb import ComplementNB

if __name__ == '__main__':

    cnb = ComplementNB()
    y = np.array([1, 0, 1, 0, 2, 1, 0, 1])
    X = np.array([[1, 2, 3, 4, 3, 3, 5, 3],
                  [9, 8, 7, 4, 3, 2, 1, 3]]).T
    d = dict(Counter(y))
    print cnb.is_fitted
    print cnb.fit(X, y).score(X, y)

