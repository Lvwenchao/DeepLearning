# write by Mrlv
# coding:utf-8

import tensorflow as tf

a = tf.random.uniform([1, 10], maxval=2, minval=-2)
a_sigmoid = tf.sigmoid(a)
print(max(a_sigmoid), min(a_sigmoid))
a_softmax = tf.nn.softmax(a_sigmoid, axis=1)
print(tf.reduce_sum(a))
print(tf.nn.tanh(a))
