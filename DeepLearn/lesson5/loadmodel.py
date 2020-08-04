# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow import keras
from DeepLearn.lesson5.customize import CustomizeModel
from tensorflow.keras import datasets, layers, Sequential, optimizers, metrics
from DeepLearn.lesson5.preprocess import preprocess,preprocess_onehot

# readData
batch_size = 128
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print('dataset:', x_train.shape, y_train.shape)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = train_db.map(preprocess_onehot()).shuffle(60000).batch(batch_size)
test_db = test_db.map(preprocess_onehot()).batch(batch_size)

# 导入模型
network = Sequential([layers.Dense(256, activation=tf.nn.relu),
                      layers.Dense(128, activation=tf.nn.relu),
                      layers.Dense(64, activation=tf.nn.relu),
                      layers.Dense(32, activation=tf.nn.relu),
                      layers.Dense(10)])

network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.load_weights("./models/my_model")

# 对存储的模型进行测试
network.evaluate(test_db)

network.save('./model.h5')
del network
