# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow import keras


class CustomizeDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(CustomizeDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class CustomizeModel(Model):
    def __init__(self, input_dim):
        super(CustomizeModel, self).__init__()
        self.ly1 = CustomizeDense(input_dim, 256)
        self.ly2 = CustomizeDense(256, 128)
        self.ly3 = CustomizeDense(128, 64)
        self.ly4 = CustomizeDense(64, 32)
        self.ly5 = CustomizeDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        out = tf.nn.relu(self.ly1(inputs))
        out = tf.nn.relu(self.ly2(out))
        out = tf.nn.relu(self.ly3(out))
        out = tf.nn.relu(self.ly4(out))
        out = self.ly5(out)

        return out


class MyDense(layers.Layer):
    def __init__(self, input_dim, ouput_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, ouput_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel
        return out


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.ly1 = MyDense(32*32*3, 256)
        self.ly2 = MyDense(256, 128)
        self.ly3 = MyDense(128, 64)
        self.ly4 = MyDense(64, 32)
        self.ly5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, [-1, 32*32*3])
        out = tf.nn.relu(self.ly1(x))
        out = tf.nn.relu(self.ly2(out))
        out = tf.nn.relu(self.ly3(out))
        out = tf.nn.relu(self.ly4(out))
        out = self.ly5(out)
        return out
