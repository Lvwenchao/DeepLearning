# write by Mrlv
# coding:utf-8
import time

import tensorflow as tf
from tensorflow.keras import optimizers
from DeepLearn.tools.customize import LSTM
from DeepLearn.lesson7 import rnnlayer

batchsz = 128
total_words = 10000
embedding_len = 100
max_words_len = 80
data_train = rnnlayer.train_data
data_test = rnnlayer.test_data


def main():
    units = 64
    epochs = 10
    model = LSTM(units, embedding_len, max_words_len, total_words)

    time0 = time.time()
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(data_train, epochs=epochs, validation_data=data_test)

    model.evaluate(data_test)
    time1 = time.time()
    print()


if __name__ == '__main__':
    main()
