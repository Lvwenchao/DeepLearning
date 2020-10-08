# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers

bn_layer = layers.BatchNormalization()
x = tf.random.normal([2, 3], mean=0, stddev=1)
out = bn_layer(x)

if __name__ == '__main__':
    print(bn_layer.trainable_variables)
    print(bn_layer.variables)
