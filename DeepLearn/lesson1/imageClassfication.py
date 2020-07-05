import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), _ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)
train_data = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)

w1 = tf.Variable(tf.random.truncated_normal([784, 256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
lr = 1e-3

for step, (x, y) in enumerate(train_data):
    x = tf.reshape(x, [-1, 28*28])
    with tf.GradientTape() as Tape:
        h1 = x @ w1 + b1
        h1 = tf.nn.relu(h1)
        h2 = h1 @ w2 + b2
        h2 = tf.nn.relu(h2)
        out = h2 @ w3 + b3
        y_oneHot = tf.one_hot(y, depth=10)
        loss = tf.square(y_oneHot - out)
        loss = tf.reduce_mean(loss)

    grads = Tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
    print(grads)
    w1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])
    w2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    w3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])
