# write by Mrlv
# coding:utf-8
import tensorflow as tf

a = tf.random.normal([128, 32, 32, 3])
c = tf.random.normal([128, 28, 28])
b = tf.reshape(a, [-1])
print(tf.reshape(c, [128, -1]).shape)
print(tf.reshape(a, [128, -1]).shape)