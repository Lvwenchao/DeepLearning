# write by Mrlv
# coding:utf-8
import tensorflow as tf

# 前向卷积
img = tf.random.normal([7, 7])
img = tf.reshape(img, [1, 7, 7, 1])

conv2d_layer = tf.keras.layers.Conv2D(1, kernel_size=3, strides=2, padding='valid')
out = conv2d_layer(img)
print(out.shape)

# 转置卷积
# 均匀添加s-1行列
transconv2d_layer = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='valid')
img2 = transconv2d_layer(out)
print(img2.shape)
