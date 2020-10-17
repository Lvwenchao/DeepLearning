import tensorflow as tf
from DeepLearn.tools import loadData
import time
import numpy as np
from tensorflow.keras import Sequential, layers, metrics, optimizers


def run():
    model_layers = [  # five unit layers
        # the first unit (tow conv and one pool)
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),

        # the second unit
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),

        # the third unit
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),

        # the forth unit
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),

        # the fifth unit
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME'),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Softmax()
    ]

    network = Sequential(model_layers)
    network.build(input_shape=[None, 32, 32, 3])
    optimizer = optimizers.Adam(1e-4)

    train_data, val_data, test_data = loadData.cifar100_data(64)
    simple = next(iter(test_data))
    print(simple[0].shape, simple[1].shape)

    for epoch in range(10):
        for step, (x, y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                y = tf.one_hot(y, depth=100)
                out = network(x)
                loss = tf.losses.categorical_crossentropy(y, out, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

            if step % 100 == 0:
                print(step, "loss", loss)

        for step, (x, y) in enumerate(val_data):
            out = network(x)
            out = tf.argmax(out, axis=-1)
            out = tf.cast(out, )
    # network = Sequential(model_layers)
    # network.build(input_shape=[None, 32, 32, 3])
    # network.summary()
    #
    # classes = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    # train_data, val_data, test_data = loadData.cifar10_data(32)
    #
    # network.compile(optimizer=optimizers.Adam(lr=0.1),
    #                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    #                 metrics=['accuracy'])
    #
    # network.fit(train_data, epochs=5, validation_data=val_data, validation_freq=1)


if __name__ == '__main__':
    run()
