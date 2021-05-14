import tensorflow as tf
from tools import loadData
from tensorflow.keras import Sequential, layers, optimizers

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
    layers.Dense(10, activation='relu'),
]


def run():
    dataset = loadData.DataSets()

    network = Sequential(model_layers)
    network.build(input_shape=[None, 32, 32, 3])
    optimizer = optimizers.Adam(1e-4)

    train_data, val_data, test_data = dataset.load_tensorflow_data('cifar10', batch_size=128, classes=10)
    simple = next(iter(train_data))
    print(simple[0].shape, simple[1].shape)

    for epoch in range(10):
        for step, (x, y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                y = tf.one_hot(y, depth=10)
                out = network(x)
                loss = tf.losses.categorical_crossentropy(y, out, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

            if step % 100 == 0:
                print(step, "loss", loss)

        correct, total = 0, 0
        for step, (x, y) in enumerate(val_data):
            out = network(x)
            # (b,10) to (b,)
            pre = tf.argmax(out, axis=-1)
            pre = tf.cast(pre, dtype=tf.int64)
            y = tf.cast(y, dtype=tf.int64)
            correct = float(tf.reduce_sum(tf.cast(tf.equal(y, pre), dtype=tf.float32)))
            total += x.shape[0]

        print("val acc:{}".format(correct / total))

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
