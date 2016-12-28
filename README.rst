
Bayes
=====

Python implementations of Naive Bayes algorithm variations with sklearn-like API.


Algorithms
----------

* Complement Naive Bayes
* Negation Naive Bayes
* Universal-set Naive Bayes

Installation
------------

.. code-block::
    pip install bayes-variants



Usage
-----

Bayes classifiers API mimics [Scikit-Learn](http://scikit-learn.org/stable/modules/classes.html) API, so usage is very simple.


.. code-block:: python

    from bayes.classifiers import ComplementNB
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    
    
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
    X_test = vectorizer.fit_transform(newsgroups_test.data)
    y_test = newsgroups_test.target
    
    # Score 
    cnb = ComplementNB()
    cnb.fit(X_train, y_train).accuracy_score(X_test, y_test)




TODO list
---------
* Weighted Complement Naive Bayes
* Locally Weighted Naive Bayes
* Selective Naive Bayes Classifier
* Add unit tests
* Add project documentation

