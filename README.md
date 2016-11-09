# Bayes
Python implementations of Naive Bayes algorithm variations with sklearn-like API.


### Complement Naive Bayes

<p align='justify'>
Complement Naive Bayes was coined as a way to tackle some Naive Bayes limitations. 
It was presented in 2003 in the paper
 <i><a href='https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf'>Tackling the Poor Assumptions of Naive Bayes Text Classifiers</a></i>
 by Rennie J. D. M. et al. 
According to the authors, classic Naive Bayes Classifier tends to be biased in case of skewed data. 
Obviously, this characteristic is just a feature of NB classifier, but in some situations it may be cause misclassification.  
The point is to compute probability for a given class <i>c</i> on all the classes except <i>c</i>.
</p>


![Image of Yaktocat](./img/eq1.png)

##### Usage

``` python
from Bayes import ComplementNB

cnb = ComplementNB()
cnb.fit(X, y).score(X, y)
```




### Negation Naive Bayes 

TODO

