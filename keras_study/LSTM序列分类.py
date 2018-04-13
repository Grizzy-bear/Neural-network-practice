# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:LSTM序列分类.py
@time:2017/11/1715:17
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import keras.utils
import numpy as np

#generate dummy data
x_train = np.random.random((100,100, 100 ,3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100,1)),num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)