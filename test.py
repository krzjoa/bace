import cPickle as pickle
import cnb as cnb
import sys
import numpy as np
sys.path.append('/home/krzysztof/Pulpit/Projekt/text2vec')
from random import shuffle
from sklearn.metrics import accuracy_score

wektory = open('/home/krzysztof/Pulpit/Projekt/App/wektory','r')
z = pickle.load(wektory)
wektory.close()
x, y = z[0], z[1]
xNew = []
yNew = []
index_shuf = range(len(x))
shuffle(index_shuf)
for i in index_shuf:
    xNew.append(x[i])
    yNew.append(y[i])
    
Xtrain = xNew[:4000]
Xtest = xNew[4000:]
Ytrain = yNew[:4000]
Ytest = yNew[4000:]

klasyfikator = cnb.ComplementNB(Xtrain, Ytrain)
y_pred =  klasyfikator.predict(Xtest)  

print accuracy_score(y_pred, Ytest) 


