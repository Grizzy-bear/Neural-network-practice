# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:itchat图片合成.py
@time:2017/11/1521:42
"""
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation

# model = Sequential([
#     Dense(32, units=784),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),
# ])

model = Sequential()
model.add(Dense(32,input_shape=784,))
model.add(Activation('relu'))

# 编译
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.compile(optimizer='rmsprop',
              loss='mse')
import keras.backend as K
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy',mean_pred()])

# 多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', # 多分类交叉熵
              metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 回归问题
model.compile(optimizer='rmsprop',
              loss='mse')

# 自定义metrics
import keras.backend as K   # 用k来表示，底层不管是TensorFlow还是theao，都可以了

def mean_pred(y_true, y_pred):  # 自定义评价函数
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])  # 使用定义的评价函数
# ### 1.4 训练
