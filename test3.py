

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print y_pred
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))


from cnb import ComplementNB

cnb = ComplementNB()
yp = cnb.fit(iris.data, iris.target).predict(iris.data)

print yp