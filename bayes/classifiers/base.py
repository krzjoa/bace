from sklearn.exceptions import NotFittedError
from abc import ABCMeta, abstractmethod
from scipy.sparse import csr_matrix
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


# Warnings


class AlphaZeroWarning(Warning):
    pass


class NotImplementedYet(Warning):
    pass


# Base Naive Bayes classifier class


class BaseNB(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.is_fitted = False
        self.classes_ = None

    @abstractmethod
    def fit(self, X, y):
        '''

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self : ComplementNB object
            Returns self.
        '''

    @abstractmethod
    def partial_fit(self, X, y, classes=None):
        """
        Incremental fit on a batch of samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        classes : array-like, shape = [n_classes], optional (default=None)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : object
             Returns self.
        """


    @abstractmethod
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Unseen samples vector
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X

        """

    @abstractmethod
    def _reset(self):
        ''''''

    # Scores

    def accuracy_score(self, X, y):
        self._check_is_fitted()
        return accuracy_score(y, self.predict(X))

    # def f1_score(self, X, y):
    #     self._check_is_fitted()
    #     return f1_score(y, self.predict(X))
    #
    # def precision_score(self, X, y):
    #     self._check_is_fitted()
    #     return precision_score(y, self.predict(X))
    #
    # def recall_score(self, X, y):
    #     self._check_is_fitted()
    #     return recall_score(y, self.predict(X))
    #
    # def roc_auc_score(self, X, y):
    #     self._check_is_fitted()
    #     return roc_auc_score(y, self.predict(X))

    # Checking params & states

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise NotFittedError

    def _check_alpha_param(self):
        if self.alpha == 0.0:
            warnings.warn('Alpha sholud not be zero. It may cause division by zero', AlphaZeroWarning)

    def _not_implemented_yet(self, message):
        warnings.warn(NotImplementedYet(message))

    def safe_mult(self, input_array, internal_array):
        if isinstance(input_array, csr_matrix):
            input_array = input_array.toarray()
        return input_array * internal_array

    def safe_matmult(self, input_array, internal_array):
        if isinstance(input_array, csr_matrix):
            input_array = input_array.toarray()
        return input_array.dot(internal_array.T)
