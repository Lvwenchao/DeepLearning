import tensorflow as tf
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)

# w1,w2,w3,b1,b2,b3
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
lr = 0.01

for epoch in range(4):
    for step, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:
            # [b,28,28]->[b,28*28]
            x = tf.reshape(x, [-1, 28 * 28])
            # h1,h2,out
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            out = h2 @ w3 + b3
            # y[b,]->[b,10]
            y = tf.one_hot(y, depth=10)
            # compute of loss
            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss', loss)

total_correct, total_num = 0, 0
for step, (x, y) in enumerate(test_data):
    # [b,28,28]->[b,784,784]
    x = tf.reshape(x, [x.shape[0], -1])
    print(x.shape, y.shape)
    with tf.GradientTape() as tape:
        # h1,h2,h3,out[b,10]
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3

        # [b,10]->[b,]

        prob = tf.nn.softmax(out, axis=1)
        pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]

acc = total_correct / total_num
print('acc:', acc)
