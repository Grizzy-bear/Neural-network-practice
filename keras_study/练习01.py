# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:练习01.py
@time:2017/11/2323:22
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np

model = Sequential()
model.add(Dense(4, 200, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100, 50, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50, 20, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20, 3, init='uniform'))
model.add(Activation('softmax'))

model.compile(loss='binary_cossentropy', optimizer='adam')
matfn = u''
data = sio.loadmat(matfn)
data = np.array(data.get('iris_train'))
trainDa = data[:80,:4]
trainB1 = data[:80,4:]
testDa = data[80:,:4]
testBl  = data[80:,4:]

model.fit(trainDa,trainB1, nb_epoch=80, batch_size=20)
print(model.evaluate(testDa, testBl))
print(model.predict_classes(testDa))
print('真实标签:\n')
print(testBl)