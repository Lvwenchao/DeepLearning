# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers

words = tf.range(5)
# customize
words = tf.random.shuffle(words)
net = layers.Embedding(10, 4)
out = net(words)
print(net.trainable_variables)
