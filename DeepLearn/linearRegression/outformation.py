# write by Mrlv
# coding:utf-8

import tensorflow as tf

a = tf.random.uniform([1, 10], maxval=2, minval=-2)

# 使得数据处于0-1之间
a_sigmoid = tf.sigmoid(a)
print(max(a_sigmoid), min(a_sigmoid))

# 使得数据内元素相加为1
a_softmax = tf.nn.softmax(a_sigmoid, axis=1)
print(a_softmax)
print(tf.reduce_sum(a_softmax))
print(tf.nn.tanh(a))
