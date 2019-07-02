from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from bace import ComplementNB, NegationNB, SelectiveNB, UniversalSetNB

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

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
        NegationNB(),
        SelectiveNB(),
        UniversalSetNB()
    ]

    def __init__(self):
        super(BenchmarkNaiveBayes, self).__init__(BenchmarkNaiveBayes.CLASSIFIERS)


if __name__ == '__main__':
    tracemalloc.start()
    from bace.utils import get_data
    X_train, y_train, X_test, y_test = get_data()
    bnb = BenchmarkNaiveBayes()
    bnb.fit(X_train, y_train)
    bnb.compare(X_test, y_test)
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot, limit=5)