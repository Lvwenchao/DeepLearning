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


def cifar10_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)
    train_data = train_data.map(preprocess)
    train_data = train_data.shuffle(50000).batch(200)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.map(preprocess)
    test_data = test_data.shuffle(10000).batch(100)

    return train_data, test_data


def minist_data(batchsize):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    print("data:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.map(preprocess).shuffle(60000).batch(batchsize)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.map(preprocess).shuffle(60000).batch(batchsize)
    return train_data, test_data


def fationmnist_data(batch_size):
    (x_train, ytrain), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data
    train_data = tf.data.Dataset.from_tensor_slices((x_train, ytrain))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_data = train_data.map(preprocess).shuffle(10000).batch(batch_size)
    test_data = test_data.map(preprocess).shuffle(x_train.shape[0]).batch(batch_size)
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

    print(dataset)


if __name__ == '__main__':
    auto_mpg_dataset()
