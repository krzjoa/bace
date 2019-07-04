# bace <img src="https://raw.githubusercontent.com/krzjoa/bace/master/img/bace-of-spades.png" align="right" width = "75px"/>
![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg) 
[![PyPI version](https://badge.fury.io/py/bace.svg)](https://badge.fury.io/py/bace) 
[![Build Status](https://travis-ci.org/rasbt/mlxtend.svg?branch=master)](https://travis-ci.org/krzjoa/Bayes) 
[![Documentation Status](https://readthedocs.org/projects/bace/badge/?version=latest)](https://bace.readthedocs.io/en/latest/?badge=latest) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


A deck of Naive Bayes algorithms with sklearn-like API.

## Algorithms
* Complement Naive Bayes
* Negation Naive Bayes
* Universal-set Naive Bayes
* Selective Naive Bayes

## Installation

You can install this module directly from GitHub repo with command:

````
python3.7 -m pip install git+https://github.com/krzjoa/bace.git
````

or as a PyPI package

````
python3.7 -m pip install bace
````

## Usage

**bace** API mimics [scikit-learn](http://scikit-learn.org/stable/modules/classes.html) API, so usage is very simple.

```` python
from bace import ComplementNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
    
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
````
