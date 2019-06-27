from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from bace import *

if __name__ == '__main__':

    cnb = ComplementNB(alpha=1., weight_normalized=True)
    mnb = MultinomialNB()
    vectorizer = CountVectorizer()

    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']

    # Train set
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          categories=categories, shuffle=True)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)

    # Test set
    newsgroups_test = fetch_20newsgroups(subset='test',
                                          categories=categories, shuffle=True)
    test_vectors = vectorizer.transform(newsgroups_test.data)


    # Comparision

    # Data
    print(train_vectors.shape)
    #print newsgroups_train.target


    # Classical MNB
    print("Classical MNB")
    mnb.fit(train_vectors, newsgroups_train.target)
    predictions = mnb.predict(test_vectors)
    print(accuracy_score(newsgroups_test.target, predictions))
    # print mnb.predict_proba(test_vectors)

    print("Complement NB")
    cnb.fit(train_vectors, newsgroups_train.target)#
    # print "Params", cnb.classes_, cnb.class_counts_, cnb.class_occurences_
    # print cnb.complement_features[:, 2]
    # print cnb.complement_features_in_classes_[2]

    # print "Log prob", cnb.predict_proba(test_vectors[:1])
    # print "Log prob", cnb.dev_predict_proba(test_vectors[:1])

    # print cnb.score(test_vectors, newsgroups_test.target)
    # print "Compl cl lp", cnb.complement_class_log_probs.values()
    # print "Compl cl lp 2", cnb.complement_class_log_probs_

    #print cnb.predict_proba(test_vectors)

    print(cnb.accuracy_score(test_vectors, newsgroups_test.target))

    # import pdb
    # pdb.set_trace()




    # y = np.array([1, 0, 1, 0, 2, 1, 0, 1])
    # X = np.array([[1, 2, 3, 4, 3, 3, 5, 3],
    #               [9, 8, 7, 4, 3, 2, 1, 3]]).T
    # d = dict(Counter(y))
    # print cnb.fit(X, y).score(X, y)

    # Get data

