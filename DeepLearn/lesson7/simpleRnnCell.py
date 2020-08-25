# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers

# 初始化状态向量
x = tf.random.normal([4, 80, 100])
x0 = x[:, 0, :]
h0 = [tf.zeros([4, 64])]
h1 = [tf.zeros_like(h0)]

# 加载sell
cell = layers.SimpleRNNCell(64)
cell2 = layers.SimpleRNNCell(64)
cell.build(input_shape=[None, 100])
cell2.build(input_shape=[None, 100])
out = tf.zeros([4, 64])
print(cell.trainable_variables)
# for x in tf.unstack(x, axis=1):
#     out0, h0 = cell(x, h0)
#     out1, h1 = cell(x, h1)
#     out = out1

# sequence_list = []
# for word in tf.unstack(x, axis=1):
#     out, h0 = cell(word, h0)
#     sequence_list.append(out)
# 
# for word in sequence_list:
#     out, h1 = cell2(word, h1)
# 
# print(out)
# 
# words = tf.random.normal([4, 80, 100])
# net = Sequential([
#     layers.SimpleRNN(64, dropout=0.5, return_sequences=True),
#     layers.SimpleRNN(64, return_sequences=True)
# ])
# out = net(words)
# print(out.shape)