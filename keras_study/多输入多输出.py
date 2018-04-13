# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:多输入多输出.py
@time:2017/11/1716:01
"""
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import numpy as np

# data_dim = 16
# timesteps = 8
# batch_size = 32
# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')
# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containg information about the entire sequence
lstm_out = LSTM(32)(x)
# 我们插入一个额外的损失，使得即使在主损失很高的情况下，LSTM和Embedding层也可以平滑的训练。
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
# 将LSTM与额外的输入数据串联起来组成输入，送入模型中
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# 最后，我们定义整个2输入，2输出的模型：
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

# main_input =np.random.random((batch_size*10, timesteps, data_dim))
# 编译
model.compile(optimizer='rmsprop',
              loss={'main_output':'binary_crossentropy', 'aux_output':'binary_crossentropy'},
              loss_weights={'main_output':1., 'aux_output':0.2})
# 输入和输出是被命名过的
# 编译完成后，我们通过传递训练数据和目标值训练该模型
model.fit({'main_input': 'headline_data', 'aux_input':'additional_data'},
          {'main_output':'labels', 'aux_output':'labels'},
          epochs=5, batch_size=32)