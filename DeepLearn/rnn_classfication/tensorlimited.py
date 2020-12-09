# write by Mrlv
# coding:utf-8

import tensorflow as tf

# clip_by_value
a = tf.random.normal([2, 2])
a = tf.clip_by_value(a, 0.4, 0.6)
print(a)

# clip_bt_norm
a = a * 5
b = tf.clip_by_norm(a, 5)
print(tf.norm(a), '\n', tf.norm(b))

# clip_by_global_norm
w1 = tf.random.normal([3, 3])
w2 = tf.random.normal([3, 3])
global_norm = tf.math.sqrt(tf.norm(w1) ** 2 + tf.norm(w2) ** 2)
(ww1, ww2), global_norm = tf.clip_by_global_norm([w1, w2], 2)
global_norm2 = tf.math.sqrt(tf.norm(ww1) ** 2, tf.norm(ww2) ** 2)
print(global_norm, '\n', global_norm2)
