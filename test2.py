from cnb import ComplementNB
from collections import Counter
from bayes_utils import get_rest
import numpy as np




if __name__ == '__main__':

    cnb = ComplementNB()
    y = np.array([1, 0, 1, 0, 2, 1, 0, 1])
    X = np.array([[1, 2, 3, 4, 3, 3, 5, 3],
                  [9, 8, 7, 4, 3, 2, 1, 3]]).T
    d = dict(Counter(y))
    print cnb.is_fitted
    cnb.fit(X, y)
    print cnb.tokens_in_classes
    #print np.where(X == 5, y)
