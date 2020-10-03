# write by Mrlv
# coding:utf-8
from abc import ABC

from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf


# 自定义层
class CustomizeDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(CustomizeDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


# 自定义模型
class MyModel(Model):
    def get_config(self):
        pass

    def __init__(self, input_dim):
        super(MyModel, self).__init__()

        self.ly1 = CustomizeDense(input_dim, 256)
        self.dp1 = layers.Dropout(0.3)
        self.ly2 = CustomizeDense(256, 128)
        self.ly3 = CustomizeDense(128, 64)
        self.ly4 = CustomizeDense(64, 32)
        # self.dp4 = layers.Dropout(0.5)
        self.ly5 = CustomizeDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 28 * 28])
        out = tf.nn.relu(self.ly1(inputs))
        out = self.dp1(out)
        out = tf.nn.relu(self.ly2(out))
        out = tf.nn.relu(self.ly3(out))
        out = tf.nn.relu(self.ly4(out))
        # out = self.dp4(out)
        out = self.ly5(out)

        return out


# CNN


# Resnet 层
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


# ResnetModel
class ResNet(Model):
    def get_config(self):
        pass

    def __init__(self, layers_dim, num_class=100):  # [2,2,2, 2]
        super(ResNet, self).__init__()
        self.start = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu'),
                                 layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                 ])
        self.layer1 = self.build_resblock(64, layers_dim[0])
        self.layer2 = self.build_resblock(128, layers_dim[1], stride=2)
        self.layer3 = self.build_resblock(256, layers_dim[2], stride=2)
        self.layer4 = self.build_resblock(512, layers_dim[3], stride=2)
        # [b,512,h,w]
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_class)

    def call(self, inputs, training=None, mask=None):
        out = self.start(inputs)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.fc(out)
        return out

    @staticmethod
    def build_resblock(filter_num, blocks, stride=1):
        res_block = Sequential()
        # may down sample
        res_block.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, stride=1))

        return res_block


# RNN层
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
        self.embedding = layers.Embedding(input_dim=total_words, output_dim=embedding_len, input_length=input_len)
        # RnnLayer
        self.rnn = Sequential([
            layers.LSTM(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.LSTM(units, dropout=0.5, unroll=True)
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


# resnet18
def resnet18():
    return ResNet([2, 2, 2, 2])
