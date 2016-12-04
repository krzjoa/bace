import cPickle as pickle
import sys

from classifiers import cnb as cnb
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from utils.bayes_utils import get_complement_matrix

x = np.array([1, 30, 4, 3, 30])

z = np.array([1, 2, 4, 30, 3, 3, 3, 3, 10, 112132])

print np.hstack((x, z))

print np.array([1, 2, 3]).dot(get_complement_matrix(3))




# lb = LabelBinarizer()
# print lb.fit_transform(x)
# print lb.classes_


# sys.path.append('/home/krzysztof/Pulpit/Projekt/text2vec')
# from random import shuffle
#
# wektory = open('/home/krzysztof/Pulpit/Projekt/App/wektory','r')
# z = pickle.load(wektory)
# wektory.close()
# x, y = z[0], z[1]
# xNew = []
# yNew = []
# index_shuf = range(len(x))
# shuffle(index_shuf)
# for i in index_shuf:
#     xNew.append(x[i])
#     yNew.append(y[i])
#
# Xtrain = xNew[:4000]
# Xtest = xNew[4000:]
# Ytrain = yNew[:4000]
# Ytest = yNew[4000:]
#
# klasyfikator = cnb.ComplementNB(Xtrain, Ytrain)
# y_pred =  klasyfikator.predict(Xtest)

#print accuracy_score(y_pred, Ytest)


