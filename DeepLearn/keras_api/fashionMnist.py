# write by Mrlv
# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras import datasets, layers, Sequential, optimizers, metrics, regularizers
from DeepLearn.tools import loadData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据预处理


# getGata and
batch_size = 128
train_data, test_data = loadData.fationmnist_data(batch_size)

db_iter = iter(train_data)
sample = next(db_iter)[0]
print(sample[0].shape, sample[1].shape)

# create model
model = Sequential([layers.Dense(256, activation=tf.nn.relu),
                    layers.Dropout(0.5),
                    layers.Dense(128, activation=tf.nn.relu),
                    layers.Dropout(0.5),
                    layers.Dense(64, activation=tf.nn.relu),
                    layers.Dropout(0.5),
                    layers.Dense(32, activation=tf.nn.relu),
                    layers.Dense(10, activation=tf.sigmoid)])
model.build(input_shape=[None, 28 * 28])
model.summary()
print(len(model.trainable_variables))

optimizer = optimizers.Adam(lr=0.001)
# create meter
loss_metric = metrics.Mean()
acc_metric = metrics.Accuracy()

# create log file
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = r'E:/pythonProject/DeepLearning/resources/logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


# 训练数据以及测试数据


def main():
    total_correct = 0
    num = 0
    for epoch in range(10):
        for step, (x, y) in enumerate(train_data):
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                out = model(x)
                y = tf.one_hot(y, depth=10)
                loss_ce = tf.losses.categorical_crossentropy(y, out)
                loss_metric.update_state(loss_ce)
                with summary_writer.as_default():
                    tf.summary.scalar("train-loss", loss_metric.result(), step=step)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(step, "loss:", loss_metric.result())
                loss_metric.reset_states()

        for step, (x, y) in enumerate(test_data):
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            prob = tf.nn.softmax(out, axis=1)
            # [b,10]->[b]
            pre = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pre, y), dtype=tf.int8))
            total_correct += correct
            num += int(x.shape[0])
            acc_metric.update_state(y, pre)

        acc = total_correct / num
        with summary_writer.as_default():
            tf.summary.scalar("test_acc", acc, epoch)
        print(epoch, 'accuracy:', acc, acc_metric.result())
        acc_metric.reset_states()


if __name__ == '__main__':
    main()
