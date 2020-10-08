# write by Mrlv
# coding:utf-8

import tensorflow as tf

x = tf.random.normal([2, 4], mean=4)
w = tf.random.normal([4, 3], stddev=0.1)
b = tf.zeros([3])
y = tf.constant([2, 1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    y = tf.one_hot(y, depth=3)
    pre = x @ w + b
    # pre = tf.nn.softmax(pre, axis=1)
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, pre, from_logits=True))

grads = tape.gradient(loss, [w, b])
print(grads[0])
