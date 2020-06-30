# write by Mrlv
# coding:utf-8

import tensorflow as tf

from tensorflow.keras import datasets, layers, optimizers, Sequential

(xs, ys), (x_val, y_val) = datasets.mnist.load_data()
print("dataSet", xs.shape, ys.shape)

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
ys = tf.convert_to_tensor(ys, dtype=tf.int32)
ys = tf.one_hot(ys, depth=10)
train_dataSet = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(200)

model = Sequential([layers.Dense(256, activation='relu'),
                    layers.Dense(256, activation='relu'),
                    layers.Dense(256, activation='relu'),
                    layers.Dense(10)])
optimizers = optimizers.SGD(learning_rate=0.01)


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataSet):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y) / x.shape[0])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizers.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, "loss:", loss.numpy())


if __name__ == '__main__':
    for epoch in range(30):
        train_epoch(epoch)
