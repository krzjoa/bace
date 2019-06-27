import unittest

from sklearn.datasets import fetch_20newsgroups

from bace import ComplementNB
from tests.data_feed import get_data


class TestCNB(unittest.TestCase):

    def test_cnb(self):
        cnb = ComplementNB()
        X_train, y_train, X_test, y_test  =  get_data()
        score = cnb.fit(X_train, y_train).accuracy_score(X_test, y_test)
        assert score > 0.80

    # def test_wcnb(self):
    #     cnb = ComplementNB(weight_normalized=True)
    #     X_train, y_train, X_test, y_test  =  get_data()
    #     score = cnb.fit(X_train, y_train).accuracy_score(X_test, y_test)
    #     assert score > 0.80

if __name__ == '__main__':
    unittest.main()

