import numpy as np
import types

# =============================================== #
#                   MATRIX UTILS                  #
# =============================================== #

def get_complement_matrix(size):
    ones = np.ones((size,))
    return np.logical_not(np.diag(ones))

# =============================================== #
#                   DOCS UTILS                    #
# =============================================== #

# This piece of code is got from http://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python


def inherit_docstring(cls):
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls


# =============================================== #
#                   SAMPLE DATA                   #
# =============================================== #

def get_data():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    vectorizer = CountVectorizer()

    # Train set
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    y_train = newsgroups_train.target

    # Test set
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)
    X_test = vectorizer.transform(newsgroups_test.data)
    y_test = newsgroups_test.target

    return X_train, y_train, X_test, y_test