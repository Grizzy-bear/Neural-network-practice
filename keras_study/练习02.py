# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:练习02.py
@time:2017/11/2523:50
"""
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1,1,28,28)
X_test = X_test.reshape(-1,1,28,28)
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential()
model.add(Convolution2D(
    nb
))
