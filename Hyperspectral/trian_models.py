# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2020/10/20 16:59
# @FileName : models.py
# @Software : PyCharm
from abc import ABC

import tensorflow as tf
from tensorflow.keras import Model, layers


class CNN(Model):
    def get_config(self):
        pass

    def __init__(self, out_channel1, out_channel2, kernal_size):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(out_channel1, kernel_size=kernal_size, strides=1)
        self.relu1 = layers.ReLU()
        self.maxpool1 = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")
        self.BN1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channel2, kernel_size=kernal_size, strides=1)
        self.relu2 = layers.ReLU()
        self.maxpool2 = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")
        self.BN2 = layers.BatchNormalization()

        self.fla = layers.Flatten()
        self.fc1 = layers.Dense(200)
        self.fc2 = layers.Dense(84)
        self.fc3 = layers.Dense(16)

    def call(self, inputs, training=None, mask=None):
        logit = self.conv1(inputs)
        logit = self.relu1(logit)
        logit = self.maxpool1(logit)
        logit = self.BN1(logit)

        logit = self.conv2(logit)
        logit = self.relu2(logit)
        logit = self.maxpool2(logit)
        logit = self.BN2(logit)

        logit = self.fla(logit)
        logit = self.fc1(logit)
        logit = self.fc2(logit)
        logit = self.fc3(logit)

        return logit
