# write by Mrlv
# coding:utf-8

import tensorflow as tf

x = tf.constant(5.)
w1 = tf.constant(1.)
w2 = tf.constant(2.)
b1 = tf.constant(3.)
b2 = tf.constant(4.)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1, w2, b1, b2])
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2

dy2_dw1 = tape.gradient(y2, [w1])[0]
print(dy2_dw1)
