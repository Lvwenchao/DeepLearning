# write by Mrlv
# coding:utf-8
import numpy as np
import tensorflow as tf

a = tf.constant([[2, 4], [1, 3], [0, 0], [0, 0]])
print(np.linalg.svd(a))
