# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2020/10/22 19:15
# @FileName : indina_cnn.py
# @Software : PyCharm
import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
from tensorflow.keras import optimizers
from Hyperspectral.patchsize import patch_size
from Hyperspectral.trian_models import CNN
import datetime


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def load_data(train_files, test_files):
    data_path = os.path.join("E:\\pythonProject\\DeepLearning", r'resources\data')

    train_data = np.array([])
    train_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    batch_size = 100

    for i in range(train_files):
        data_set = sio.loadmat(os.path.join(data_path, "train_" + str(patch_size) +
                                            '_' + str(i + 1) + '.mat'))

        if i == 0:
            train_data = data_set['train_data']
            train_label = data_set['train_label']
        else:
            train_data = np.concatenate((train_data, data_set['train_data']),
                                        axis=0)
            train_label = np.concatenate((train_label, data_set['train_label']),
                                         axis=1)

    for i in range(test_files):
        data_set = sio.loadmat(os.path.join(data_path, "test_" + str(patch_size) +
                                            '_' + str(i + 1) + '.mat'))

        if i == 0:
            test_data = data_set['test_data']
            test_label = data_set['test_label']
        else:
            test_data = np.concatenate((test_data, data_set['test_data']),
                                       axis=0)
            test_label = np.concatenate((test_label, data_set['test_label']),
                                        axis=1)

    train_data = np.transpose(train_data, (0, 2, 3, 1))
    train_label = np.squeeze(np.transpose(train_label))
    test_data = np.transpose(test_data, (0, 2, 3, 1))
    test_label = np.squeeze(np.transpose(test_label))

    train_db = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_db = train_db.map(preprocess).shuffle(1000).batch(batch_size)
    test_db = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_db = test_db.map(preprocess).shuffle(1000).batch(batch_size)

    return train_db, test_db


def train():
    train_db, test_db = load_data(8, 6)
    sample = next(iter(train_db))
    print(sample[0].shape, sample[1].shape)

    network = CNN(500, 100, 3)
    network.build(input_shape=(None, 11, 11, 220))
    network.summary()

    optimizer = optimizers.Adam(0.001)
    loss_metric = tf.metrics.Mean()
    acc_metric = tf.metrics.Accuracy()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = r'E:/pythonProject/DeepLearning/resources/logs/indian_' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    loss_step = 0
    acc_step = 0
    total_cor = 0
    total_num = 0

    for epoch in range(150):
        for step, (x, y) in enumerate(train_db):
            y = tf.one_hot(y, depth=16)
            with tf.GradientTape() as tape:
                logit = network(x)
                logit = tf.nn.softmax(logit, axis=0)
                loss = tf.losses.categorical_crossentropy(y, logit, from_logits=True)
                loss = tf.reduce_mean(loss)
                loss_metric.update_state(loss)

            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

            if loss_step % 20 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar("train-loss", loss_metric.result(), step=loss_step)

                loss_step += 1
                print('Step %d : loss = %.2f' % (loss_step, float(loss_metric.result())))
                loss_metric.reset_states()

        for step, (x, y) in enumerate(test_db):
            y = tf.cast(y, dtype=tf.int32)
            acc_step += 1
            prob = network(x)
            prob = tf.nn.softmax(prob, axis=1)
            pre = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.cast(tf.equal(pre, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_cor += correct
            total_num += int(x.shape[0])
            acc_metric.update_state(y, pre)

        acc = total_cor / total_num

        with summary_writer.as_default():
            tf.summary.scalar("accuracy", acc, step=epoch)
        print("epoch %d : accuracy = %.3f" % (epoch, float(acc_metric.result())))

    network.save_weights(r'E:\pythonProject\DeepLearning\resources\models\indian\indian_cnn')


if __name__ == '__main__':
    train()
