# write by Mrlv
# coding:utf-8
from abc import ABC
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf


# customize layers
class CustomizeDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(CustomizeDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


# customize model
class MyModel(Model):
    def get_config(self):
        pass

    def __init__(self, input_dim):
        super(MyModel, self).__init__()

        self.fla = layers.Flatten()
        self.ly1 = CustomizeDense(input_dim, 256)
        self.dp1 = layers.Dropout(0.3)
        self.ly2 = CustomizeDense(256, 128)
        self.ly3 = CustomizeDense(128, 64)
        self.ly4 = CustomizeDense(64, 32)
        # self.dp4 = layers.Dropout(0.5)
        self.ly5 = CustomizeDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        inputs = self.fla(inputs)
        out = tf.nn.relu(self.ly1(inputs))
        out = self.dp1(out)
        out = tf.nn.relu(self.ly2(out))
        out = tf.nn.relu(self.ly3(out))
        out = tf.nn.relu(self.ly4(out))
        # out = self.dp4(out)
        out = self.ly5(out)

        return out


# CNN LetNet
class LetNet(Model, ABC):
    def __init__(self):
        super(LetNet, self).__init__()
        tf.nn.conv2d()
        self.conv2d_1 = layers.Conv2D(6, kernel_size=5, strides=1)
        self.bn1 = layers.BatchNormalization()
        self.pool_max1 = layers.MaxPool2D(pool_size=2, strides=2)
        self.relu1 = layers.ReLU()
        self.conv2d_2 = layers.Conv2D(16, kernel_size=5, strides=1)
        self.bn2 = layers.BatchNormalization()
        self.pool_max2 = layers.MaxPool2D(pool_size=2, strides=2)
        self.relu2 = layers.ReLU()
        self.fla = layers.Flatten()

        # 全连接层
        self.fc1 = layers.Dense(120, activation='relu')
        self.fc2 = layers.Dense(84, activation='relu')
        self.fc3 = layers.Dense(10)
        self.softmax = layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        # [b,28,28,1]->[b,24,24,6]
        out = self.conv2d_1(inputs)
        out = self.bn1(out)
        # [b,24,24,6]->[b,12,12,6]
        out = self.pool_max1(out)
        out = self.relu1(out)
        # [b,12,12,6] -> [b,10,10,16]
        out = self.conv2d_2(out)
        out = self.bn2(out)
        # [b,10,10,16]->[b,5,5,16]
        out = self.pool_max2(out)
        out = self.relu2(out)
        # [b,5,5,16] -> [b,400]
        out = self.fla(out)

        # [b,400] -> [n,120] -> [b,84] -> [b,10]
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        out = self.softmax(out)
        return out


"""
 RNN 实现
"""


# resbasicBlock
class BasicBlock(layers.Layer):
    """
    filter_dim
    stride
    """
    def __init__(self, filter_dim, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_dim, kernel_size=(3, 3), strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filter_dim, kernel_size=(3, 3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        # 残差层
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_dim, kernel_size=(3, 3), strides=stride, padding="same"))
        elif stride == 1:
            self.downsample = lambda x: x
        else:
            raise Exception("inputshape channel can't match outputshape")

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        inputs = self.downsample(inputs)

        out = tf.add(out, inputs)
        return out


def build_block(filter_dim, blocks, stride=1):
    res_block = Sequential()
    res_block.add(BasicBlock(filter_dim=filter_dim, stride=stride))
    for i in range(1, blocks):
        res_block.add(BasicBlock(filter_dim=filter_dim, stride=1))

    return res_block


# RNN models
class ResNet(Model, ABC):
    def __init__(self, layer_dims, classes):
        super(ResNet, self).__init__()

        self.start = Sequential([layers.Conv2D(64, (3, 3), strides=1, padding="same", ),
                                 layers.BatchNormalization(),
                                 layers.ReLU(),
                                 layers.MaxPool2D(pool_size=(2, 2), strides=2)])

        self.layer1 = build_block(64, layer_dims[0])
        self.layer2 = build_block(128, layer_dims[1], stride=2)
        self.layer3 = build_block(256, layer_dims[2], stride=2)
        self.layer4 = build_block(512, layer_dims[3], stride=2)

        self.fla = layers.Flatten()

        self.fc = layers.Dense(classes)

    def call(self, inputs, training=None, mask=None):
        out = self.start(inputs)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.fla(out)
        out = self.fc(out)

        return out


class RNN(Model, ABC):
    def __init__(self, units, embedding_len, input_len, total_words):
        """

        :param units: simrnn 向量大小
        :param embedding_len: 单词的向量长度
        :param input_len: 输入每个单个时间序列的长度
        :param total_words: 句子数
        """
        super(RNN, self).__init__()
        # embedding
        self.embedding = layers.Embedding(input_dim=total_words, output_dim=embedding_len, input_length=input_len)

        # RnnLayer
        self.rnn = Sequential([
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(units, dropout=0.5, unroll=True)
        ])

        # fullConnection
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        out = inputs
        # [b,80]-[b,80,100]
        out = self.embedding(out)
        # [b,80,100]->[b,64]
        out = self.rnn(out)
        # [b,64]->[b,1]
        out = self.fc(out)
        prob = tf.sigmoid(out)

        return prob


class LSTM(Model, ABC):
    def __init__(self, units, embedding_len, input_len, total_words):
        """

        :param units: simrnn 向量大小
        :param embedding_len: 单词的向量长度
        :param input_len: 输入每个单个时间序列的长度
        :param total_words: 句子数
        """
        super(LSTM, self).__init__()
        # embedding
        self.embedding = layers.Embedding(input_dim=total_words,
                                          output_dim=embedding_len,
                                          input_length=input_len)

        # RnnLayer
        self.rnn = Sequential([
            layers.LSTM(units,
                        dropout=0.5,
                        return_sequences=True,
                        unroll=True),
            layers.LSTM(units,
                        ropout=0.5,
                        unroll=True)
        ])

        # fullConnection
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        out = inputs
        # [b,80]-[b,80,100]
        out = self.embedding(out)
        # [b,80,100]->[b,64]
        out = self.rnn(out)
        # [b,64]->[b,1]
        out = self.fc(out)
        prob = tf.sigmoid(out)

        return prob


# 自编码器
class AE(Model):
    def __init__(self, hdim):
        super(AE, self).__init__()

        self.encoder = Sequential([
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(hdim),
        ])

        self.decoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None, mask=None):
        # [b,784]->[b,20]
        h = self.encoder(inputs)
        # [b,20]->[b,784]
        x_hat = self.decoder(h)

        return x_hat

    def get_config(self):
        pass


class AEConv(Model):
    def __init__(self):
        super(AEConv, self).__init__()
        self.encoder = Sequential([
            layers.C
        ])

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass


# resnet18
def resnet18(classes):
    return ResNet(layer_dims=[2, 2, 2, 2], classes=classes)
