# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:类似VGG的卷积神经网络.py
@time:2017/11/1714:51
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from  keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD

#generate dummy data
x_train = np.random.random((100,100, 100 ,3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100,1)),num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input :100x100 image with 3 channels ->(100,100,3) tensor
# this applies 32 convolution filters of size 3x3 each
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) #压扁平准备全连接
#全连接
model.add(Dense(256, activation='relu')) #添加256节点的全连接, #激活
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#设定求解器
sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)

