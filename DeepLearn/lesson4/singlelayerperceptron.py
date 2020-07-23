# write by Mrlv
# coding:utf-8
import tensorflow as tf

input_data = tf.random.normal([1, 3])
w = tf.random.normal([3, 1])
b = tf.zeros([1])
y = tf.constant([1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    logits = input_data @ w + b
    logits = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.losses.MSE(y, logits))

grads = tape.gradient(loss, [w, b])
print(grads[1])
