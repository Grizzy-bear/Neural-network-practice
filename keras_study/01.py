# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:01.py
@time:2017/12/1723:59
"""
from keras.models import load_model
from keras.models import Sequential
from keras.applications import VGG16

from keras.layers import Dense
from keras.optimizers import SGD
# model = Sequential()
# # model.add(Dense(output_dim=1, input_dim=1))
# # model.compile(loss='mse', optimizer=SGD)
# model.save('my_model.h5')
model = VGG16()
print(model.summary())