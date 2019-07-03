from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from bace import ComplementNB, NegationNB, SelectiveNB, UniversalSetNB
import sklearn as skl


class Benchmark(BaseEstimator):
    '''

    scikit-learn like classifiers benchmark

    Parameters
    ----------
    classifiers: list of sklearn.base.BaseEstimator
        List of sklearn classifiers
    verbose: bool
        Print training details
    '''

    def __init__(self, classifiers, verbose=False):
        self.classifiers = classifiers
        self.verbose = verbose


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

            if self.verbose:
                print("{} fitted".format(clf.__class__.__name__))

        return self

    def predict(self, X):
        return [clf.predict(X) for clf in self.classifiers]

    def compare(self, X, y, metrics={'Accuracy': accuracy_score}):
        '''

        Compare predictions of multiple classifiers

        Parameters
        ----------
        X: numpy.ndarray
            Features
        y: numpy.ndarray
            Targets
        metrics: dict of callable
            List of metric functions

        '''

        for clf in self.classifiers:

            print(clf.__class__.__name__)
            predictions = clf.predict(X)

            for metric_name, metric_fun in metrics.items():
                print("{}: {}".format(
                    metric_name,
                    metric_fun(y, predictions)
                ))


class BenchmarkNaiveBayes(Benchmark):

    CLASSIFIERS = [
        MultinomialNB(),
        ComplementNB(weight_normalized=True),
        ComplementNB(),
        skl.naive_bayes.ComplementNB(),
        skl.naive_bayes.ComplementNB(norm=True),
        NegationNB(),
        SelectiveNB(),
        UniversalSetNB()
    ]

    def __init__(self):
        super(BenchmarkNaiveBayes, self).__init__(BenchmarkNaiveBayes.CLASSIFIERS)


if __name__ == '__main__':
    from bace.utils import get_data
    X_train, y_train, X_test, y_test = get_data()
    bnb = BenchmarkNaiveBayes()
    bnb.fit(X_train, y_train)
    bnb.compare(X_test, y_test)