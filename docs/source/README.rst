bace
====
.. image:: https://img.shields.io/badge/python-3.7-blue.svg
:target: https://bace.readthedocs.io/en/latest/?badge=latest
:alt: Python version
.. image:: https://readthedocs.org/projects/bace/badge/?version=latest
:target: https://badge.fury.io/py/bace
:alt: PyPI
.. image:: https://travis-ci.org/rasbt/mlxtend.svg?branch=master
:target: https://travis-ci.org/krzjoa/Bayes
:alt: CI
.. image:: https://readthedocs.org/projects/bace/badge/?version=latest
:target: https://bace.readthedocs.io/en/latest/?badge=latest
:alt: Documentation Status
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
:target: https://opensource.org/licenses/MIT)
:alt: Licence

A deck of Naive Bayes algorithms with sklearn-like API.

Algorithms
----------
* Complement Naive Bayes
* Negation Naive Bayes
* Universal-set Naive Bayes
* Selective Naive Bayes
Installation
------------

You can install this module directly from GitHub repo with command:

.. code-block::

   python3.7 -m pip install git+https://github.com/krzjoa/bace.git

or as a PyPI package

.. code-block::

   python3.7 -m pip install bace

Usage
-----

Bayes classifiers API mimics `Scikit-Learn <http://scikit-learn.org/stable/modules/classes.html>`_ API, so usage is very simple.

.. code-block:: python

   from bace import ComplementNB
   from sklearn.datasets import fetch_20newsgroups
   from sklearn.feature_extraction.text import CountVectorizer

   # Train set
   newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
   X_train = vectorizer.fit_transform(newsgroups_train.data)
   y_train = newsgroups_train.target

   # Test set
   newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)
   X_test = vectorizer.fit_transform(newsgroups_test.data)
   y_test = newsgroups_test.target

   # Score
   cnb = ComplementNB()
   cnb.fit(X_train, y_train).accuracy_score(X_test, y_test)