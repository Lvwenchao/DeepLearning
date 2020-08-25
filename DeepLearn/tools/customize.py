# write by Mrlv
# coding:utf-8
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf


# 自定义层
class CustomizeDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(CustomizeDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


# 自定义模型
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
        self.ly1 = MyDense(32 * 32 * 3, 256)
        self.ly2 = MyDense(256, 128)
        self.ly3 = MyDense(128, 64)
        self.ly4 = MyDense(64, 32)
        self.ly5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        out = tf.nn.relu(self.ly1(x))
        out = tf.nn.relu(self.ly2(out))
        out = tf.nn.relu(self.ly3(out))
        out = tf.nn.relu(self.ly4(out))
        out = self.ly5(out)
        return out


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

    def build_resblock(self, filter_num, blocks, stride=1):
        res_block = Sequential()
        # may down sample
        res_block.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, stride=1))

        return res_block


# resnet18
def resnet18():
    return ResNet([2, 2, 2, 2])


# RNN层
class RNN(Model):
    def __init__(self, units, embedding_len, input_len, total_words):
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
