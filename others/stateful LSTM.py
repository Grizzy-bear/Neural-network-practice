# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:stateful LSTM.py
@time:2017/11/1715:32
"""
from keras.models import Sequential
from keras.layers import LSTM,Dense
import numpy as np
data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# generate dummy training data
x_train = np.random.random((batch_size*10, timesteps, data_dim))
y_train = np.random.random((batch_size*10, num_classes))

# generate dummy validation data
x_val = np.random.random((batch_size*3, timesteps, data_dim))
y_val = np.random.random((batch_size*3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
