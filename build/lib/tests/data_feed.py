
def get_data():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    vectorizer = CountVectorizer()

    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']

    # Train set
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          categories=categories, shuffle=True)
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    y_train = newsgroups_train.target

    # Test set
    newsgroups_test = fetch_20newsgroups(subset='test',
                                         categories=categories, shuffle=True)
    X_test = vectorizer.transform(newsgroups_test.data)
    y_test = newsgroups_test.target

    return X_train, y_train, X_test, y_test