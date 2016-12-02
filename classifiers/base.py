from sklearn.exceptions import NotFittedError
from abc import ABCMeta, abstractmethod
from scipy.sparse import csr_matrix

class BaseNB(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        ''''''

    @abstractmethod
    def partial_fit(self, X, y):
            ''''''

    @abstractmethod
    def predict(self, X):
        ''''''

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise NotFittedError
    #
    def safe_mult(self, input_array, internal_array):
        if isinstance(input_array, csr_matrix):
            input_array = input_array.toarray()
        return input_array * internal_array

