# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import datasets


def data_preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


def cifar10_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)
    train_data = train_data.map(data_preprocess)
    train_data = train_data.shuffle(50000).batch(200)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.map(data_preprocess)
    test_data = test_data.shuffle(10000).batch(100)

    return train_data, test_data


def minist_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)
    train_data = train_data.map(data_preprocess)
    train_data = train_data.shuffle(60000).batch(200)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.map(data_preprocess)
    test_data = test_data.shuffle(10000).batch(100)
    return train_data, test_data


def fationmnist_data():
    (x_train, ytrain), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data
