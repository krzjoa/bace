import unittest
from collections import Counter
import numpy as np
from classifiers.cnb import ComplementNB
from sklearn.datasets import fetch_20newsgroups


class TestCNB(unittest.TestCase):

    def test_cnb(self):
        cnb = ComplementNB()
        news = fetch_20newsgroups()
        print news.__dict__
        # y = np.array([1, 0, 1, 0, 2, 1, 0, 1])
        # X = np.array([[1, 2, 3, 4, 3, 3, 5, 3],
        #               [9, 8, 7, 4, 3, 2, 1, 3]]).T
        # d = dict(Counter(y))
        # print cnb.fit(X, y).score(X, y)

        # Get data


if __name__ == '__main__':
    unittest.main()

