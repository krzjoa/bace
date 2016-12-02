from collections import Counter
import numpy as np
from classifiers.cnb import ComplementNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    cnb = ComplementNB()
    mnb = MultinomialNB()
    vectorizer = TfidfVectorizer()

    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']

    # Train set
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          categories=categories)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)

    # Test set
    newsgroups_test = fetch_20newsgroups(subset='test',
                                          categories=categories)
    test_vectors = vectorizer.transform(newsgroups_test.data)


    # Comparision

    # Data
    print train_vectors.shape
    print test_vectors.shape


    # Classical MNB
    print "Classical MNB"
    mnb.fit(train_vectors, newsgroups_train.target)
    predictions = mnb.predict(test_vectors)
    print accuracy_score(newsgroups_test.target, predictions)

    print "Complement NB"
    cnb.fit(train_vectors, newsgroups_train.target)#
    print cnb.score(test_vectors, newsgroups_test.target)

    # import pdb
    # pdb.set_trace()




    # y = np.array([1, 0, 1, 0, 2, 1, 0, 1])
    # X = np.array([[1, 2, 3, 4, 3, 3, 5, 3],
    #               [9, 8, 7, 4, 3, 2, 1, 3]]).T
    # d = dict(Counter(y))
    # print cnb.fit(X, y).score(X, y)

    # Get data

