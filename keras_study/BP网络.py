# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:BP网络.py
@time:2017/11/1819:09
"""
import pandas as pd
inputfile = '02.xlsx'
outputfile = 'output.xlsx'
modelfile = 'modelweight.model'
data = pd.read_excel(inputfile, index='Date', sheetname=0)

feature = ['F1', 'F2', 'F3', 'F4','F5','F6','F7','F8']
label = ['F9']
data_train = data.loc[range(0,520)].copy()

data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std
x_train = data_train[feature].as_matrix()
y_train = data_train[label].as_matrix()

from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform' ))
model.add(Activation('relu'))
model.add(Dense(1, input_dim=20))
model.compile(loss='mean_squared_error', optimizer='adma')
model.fit(x_train, y_train, nb_epoch=1000, batch_size=6)
model.save_weights(modelfile)

x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
data[u'F9_pred'] = model.predict(x)*data_std['F9']  +data_mean['F9']

data.to_excel(outputfile)

import matplotlib.pyplot as plt
p = data[['L1','L1_pred']].plot(subplots = True, style=['b-o','r-*'])
plt.show()