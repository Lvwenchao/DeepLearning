# write by Mrlv
# coding:utf-8
import tensorflow as tf

y = tf.range(4)
y = tf.one_hot(y, depth=5)
out = tf.random.normal([4, 5])
loss1 = tf.reduce_mean(tf.square(y - out))
loss2 = tf.square(tf.norm(y - out)) / (y.shape[0] * y.shape[1])
loss3 = tf.reduce_mean(tf.losses.MSE(y, out))
print(float(loss1), loss2, loss3)
y = tf.fill([4], 0.25)
print(y * tf.math.log(y) / tf.math.log(2.))
y_entropy = -tf.reduce_sum(y * tf.math.log(y) / tf.math.log(2.))
print(y_entropy)
y_crossentropy = tf.losses.categorical_crossentropy([1, 0, 0, 0], y)
print(y_crossentropy)
criteon = tf.losses.BinaryCrossentropy()
print(criteon([1, 0], [0.9, 0.1]))
print(tf.losses.binary_crossentropy())
