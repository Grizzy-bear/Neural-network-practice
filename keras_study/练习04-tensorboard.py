# encoding: utf-8
"""
@version:3.6
@author:lamplight
@file:练习04-tensorboard.py
@time:2017/12/50:3cd
"""
import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
sess.run(e)

writer = tf.summary.FileWriter("F:/1AFILE/python/Keras/graph", tf.get_default_graph())
writer.close()