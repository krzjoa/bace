# from sklearn import datasets
# iris = datasets.load_digits()
#
# print iris.data[:10]
# print iris.target[:20]
#
# from sklearn.naive_bayes import MultinomialNB
# gnb = MultinomialNB()
# y_pred = gnb.fit(iris.data, iris.target).score(iris.data, iris.target)
# print y_pred
# print("Number of mislabeled points out of a total %d points : %d"
#       % (iris.data.shape[0],(iris.target != y_pred).sum()))
#
#
# from bayes.classifiers import ComplementNB
#
# cnb = ComplementNB()
# yp = cnb.fit(iris.data, iris.target).score(iris.data, iris.target)
#
# print yp