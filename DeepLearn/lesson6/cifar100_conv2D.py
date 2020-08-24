# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import datasets, optimizers, layers, Sequential
from DeepLearn.tools.preprocess import preprocess

conv_layers = [  # five unit layers
    # the first unit (tow conv and one pool)
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # the second unit
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # the third unit
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # the forth unit
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # the fifth unit
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")
]


def main():
    conv_net = Sequential(conv_layers)
    conv_net.build(input_shape=[None, 32, 32, 3])

    fac_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=tf.nn.relu)
    ])
    fac_net.build(input_shape=[None, 512])
    variables = fac_net.trainable_variables + conv_net.trainable_variables

    batch_size = 50
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    y_train = tf.squeeze(y_train)
    y_test = tf.squeeze(y_test)
    print("data:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    train_date = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_date = train_date.shuffle(10000).map(preprocess).batch(batch_size)
    test_data = test_data.shuffle(10000).map(preprocess).batch(batch_size)

    sample = next(iter(train_date))
    print(sample[0].shape, sample[1].shape, tf.reduce_max(sample[1]), tf.reduce_min(sample[1]))

    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(50):
        for step, (x_train, y_train) in enumerate(train_date):
            # [b,32,32,3]->[b,1,1,52]
            with tf.GradientTape() as tape:
                out = conv_net(x_train)
                out = tf.reshape(out, [-1, 512])
                # [b,512]->[b,100]
                logits = fac_net(out)
                y_onehot = tf.one_hot(y_train, depth=100)

                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(step, "loss:", float(loss))

        # test
        correct_total, num_total = 0, 0
        for step, (x, y) in enumerate(test_data):
            # [b,32,32,3}->[b,1,1,512]
            out = conv_net(x)
            # [b,1,1,512]->[b,512]
            out = tf.reshape(out, [-1, 512])
            # [b,512]->[b,100]
            logits = fac_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            # [b,100]->[b,]
            pre = tf.argmax(prob, axis=1)
            pre = tf.cast(pre, dtype=tf.int32)
            # compute accrucy
            correct = tf.cast(tf.equal(y, pre), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            correct_total += int(correct)
            num_total += x.shape[0]

        accuracy = float(correct_total / num_total)
        print(epoch, "acc", accuracy)


if __name__ == '__main__':
    main()
