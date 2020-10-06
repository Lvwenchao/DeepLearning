# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import pandas as pd


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


def cifar10_data(batch_size):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = tf.squeeze(y_train)
    y_test = tf.squeeze(y_test)
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    x_train = tf.reshape(x_train, [-1, 32 * 32 * 3])
    x_test = tf.reshape(x_test, [-1, 32 * 32 * 3])
    print("data:", 'train', x_train.shape, y_train.shape, 'test:', x_test.shape, y_test.shape)
    x_train, x_val = tf.split(x_train, axis=0, num_or_size_splits=[40000, 10000])
    y_train, y_val = tf.split(y_train, axis=0, num_or_size_splits=[40000, 10000])
    train_date = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_date = train_date.map(preprocess).shuffle(40000).batch(batch_size)
    val_data = val_data.map(preprocess).shuffle(10000).batch(batch_size)
    test_data = test_data.map(preprocess).shuffle(10000).batch(batch_size)
    return train_date, val_data, test_data


def cifar_100_data(batch_size):
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    print('data:', x_train.shape, y_train.shape)
    y


def minist_data(batch_size):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    x_train, x_val = tf.split(x_train, num_or_size_splits=[50000, 10000])
    y_train, y_val = tf.split(y_train, num_or_size_splits=[50000, 10000])
    print("data:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.map(preprocess).shuffle(50000).batch(batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.map(preprocess).shuffle(10000).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.map(preprocess).shuffle(10000).batch(batch_size)
    return train_data, val_data, test_data


def fationmnist_data(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    y_train = tf.squeeze(y_train)
    y_test = tf.squeeze(y_test)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_data = train_data.map(preprocess).shuffle(60000).batch(batch_size)
    test_data = test_data.map(preprocess).shuffle(60000).batch(batch_size)
    return train_data, test_data


def auto_mpg_dataset():
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    cocolumn_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year',
                      'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=cocolumn_names, sep=' ', skipinitialspace=True, na_values=0,
                              comment='\t')
    dataset = raw_dataset.copy()  # 查看部分数据
    dataset.isna().sum()  # 统计空白数据
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    dataset.tail()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_label = train_dataset.pop('MPG')
    test_label = train_dataset.pop('MPG')
    return (train_dataset, test_dataset), (train_label, test_label)
