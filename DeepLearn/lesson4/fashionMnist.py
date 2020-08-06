# write by Mrlv
# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras import datasets, layers, Sequential, optimizers, metrics, regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 数据预处理
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# getGata and
batch_size = 128
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = train_db.map(preprocess).shuffle(10000).batch(batch_size)
test_db = test_db.map(preprocess).shuffle(10000).batch(batch_size)

db_iter = iter(train_db)
sample = next(db_iter)[0]
sample_image = tf.reshape(x_train[:25, :, :], [-1, 28, 28, 1])
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
log_dir = "logs/" + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default():
    tf.summary.image("Train_sample", sample_image, step=0)


# 训练数据以及测试数据
def main():
    total_correct, total_num = 0, 0
    for epoch in range(10):
        for step, (x, y) in enumerate(train_db):
            x = tf.reshape(x, [-1, 28 * 28])
            # [b,]->[b,10]
            y = tf.one_hot(y, depth=10)
            with tf.GradientTape() as tape:
                out = model(x)  # [b,784]->[b,10]
                # mse
                loss_mse = tf.reduce_mean(tf.losses.MSE(y, out))
                # cross_entrope
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y, out, from_logits=True))
                loss_regularization = []
                for p in model.trainable_variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_mean(tf.stack(loss_regularization))
                loss_ce = loss_ce + 0.001 * loss_regularization
                loss_metric.update_state(loss_ce)
            grads = tape.gradient(loss_ce, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(step, "loss:", loss_metric.result().numpy())
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss_ce), step=epoch)

        # model test
        for step, (x, y) in enumerate(test_db):
            x = tf.reshape(x, [-1, 28 * 28])
            # [b,784]->[b,10]
            prob = model(x)
            # [b,10]->[b]
            prob = tf.nn.softmax(prob, axis=1)
            pre = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pre, y), dtype=tf.int32))
            total_correct += int(correct)
            total_num += x.shape[0]

            # 用 tensorflow提供的方法代替上面效果
            acc_metric.update_state(y, pre)

        acc = total_correct / total_num
        print(epoch, "accuracy:", acc, acc_metric.result().numpy())
        with summary_writer.as_default():
            tf.summary.scalar('test-acc', float(acc), step=epoch)


if __name__ == '__main__':
    main()
