# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:练习_tensorflow.py
@time:2017/12/1012:24
"""
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

A = tf.placeholder(tf.float32, name="A")
B = tf.placeholder(tf.float32, name='B')

cal_op = tf.add(A,B, name="addition")


session = tf.Session()
result = session.run(cal_op, feed_dict={A:[10], B:[32]})

print(result)
session.close()