# Bayes
Python implementations of Naive Bayes algorithm variations with sklearn-like API.

## Algorithms

#### Complement Naive Bayes

<p align='justify'>
Complement Naive Bayes was coined as a way to tackle some Naive Bayes limitations. 
It was presented in 2003 in the paper
 <i><a href='https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf'>Tackling the Poor Assumptions of Naive Bayes Text Classifiers</a></i>
 by Rennie J. D. M. et al. 
</p>

#### Weight-normalized Complement Naive Bayes

This method is a variant of Complement Naive Bayes with weigths normalization.
<i><a href='https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf'>Tackling the Poor Assumptions of Naive Bayes Text Classifiers</a></i>


![Equation 1](./img/eq1.png)

#### Negation Naive Bayes

<i><a href='http://www.aclweb.org/anthology/R11-1083.pdf'>Negation Naive Bayes for Categorization of Product Pages on the Web</a></i>



## Usage

Bayes classifiers API mimics [Scikit-Learn](http://scikit-learn.org/stable/modules/classes.html) API, so usage is very simple.


``` python
from Bayes.classifiers import ComplementNB
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
```




## TODO list
* Weighted Complement Naive Bayes
* Negation Naive Bayes
* Locally Weighted Naive Bayes
* Universal-set Naive Bayes Classifier
* Selective Naive Bayes Classifier
* Add project Wiki
