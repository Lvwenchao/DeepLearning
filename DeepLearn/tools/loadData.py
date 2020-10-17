# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split


# 加载自定义数据集
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


class DataSets(object):
    def __init__(self):
        self.data_dict = {
            'mnist': datasets.mnist.load_data,
            'fashon_mnist': datasets.fashion_mnist.load_data,
            'cifar10': datasets.cifar10.load_data,
            'cifar100': datasets.cifar100.load_data,
        }

    def load_data(self, dataset_name, batch_size, classes):
        """
        :param self:
        :param classes: 所含内别
        :param dataset_name: 数据集名称
        :param batch_size:  分割大小
        :return: 训练、验证、测试数据
        """
        if dataset_name not in self.data_dict.keys():
            raise Exception('no {} dataset'.format(dataset_name))
        else:
            train_size = 0.8
            (x_train, y_train), (x_test, y_test) = self.data_dict[dataset_name]()
            y_train = tf.squeeze(y_train)
            y_test = tf.squeeze(y_test)
            y_train = tf.one_hot(y_train, depth=classes)
            y_test = tf.one_hot(y_test, depth=classes)
            x_train, x_val = train_test_split(x_train, train_size=0.8)
            y_train, y_val = train_test_split(y_train.numpy(), train_size=0.8)
            train_date = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            train_date = train_date.map(preprocess).shuffle(10000).batch(batch_size)
            val_data = val_data.map(preprocess).shuffle(10000).batch(batch_size)
            test_data = test_data.map(preprocess).shuffle(10000).batch(batch_size)
            return train_date, val_data, test_data
