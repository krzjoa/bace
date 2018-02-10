from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from bayes.classifiers import ComplementNB, NegationNB, SelectiveNB, UniversalSetNB


class Benchmark(object):
    '''

    scikit-learn like classifiers benchmark

    Parameters
    ----------
    classifiers: list of sklearn.base.BaseEstimator
        List of sklearn classifiers

    '''

    def __init__(self, classifiers):
        self.classifiers = classifiers


    def fit(self, X, y):
        '''

        Fit several classifiers

        Parameters
        ----------
        X: numpy.ndarray

        y: numpy.ndarray
           Labels

        Returns
        -------

        '''

        for clf in self.classifiers:
            clf.fit(X, y)
        return self

    def predict(self, X):
        return [clf.predict(X) for clf in self.classifiers]

    def compare(self, X, y, metrics={'F1': f1_score}):

        for clf in self.classifiers:
            pass



class BenchmarkNaiveBayes(Benchmark):

    CLASSIFIERS = [
        ComplementNB(),
        NegationNB(),
        SelectiveNB(),
        UniversalSetNB()
    ]

    def __init__(self):
        super(BenchmarkNaiveBayes, self).__init__(BenchmarkNaiveBayes.CLASSIFIERS)


if __name__ == '__main__':
    from utils import get_data
    X_train, y_train, X_test, y_test = get_data()
    bnb = BenchmarkNaiveBayes()