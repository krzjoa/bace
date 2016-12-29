from sklearn.naive_bayes import MultinomialNB

from bayes.classifiers import *


class BenchmarkAll(object):

    def __init__(self):
        self.classifiers = self._init_classifiers()

    def _init_classifiers(self):
        mnb = MultinomialNB()
        cnb = ComplementNB()
        nnb = NegationNB()
        unb = UniversalSetNB()
        snb = SelectiveNB()
        return [mnb, cnb, nnb, unb, snb]

    def fit(self, X, y):

        for clf in self.classifiers:
            clf.fit(X, y)

        return self



    def accuracy_score(self):
        pass


