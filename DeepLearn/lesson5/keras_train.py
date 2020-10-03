# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import datasets, optimizers
from DeepLearn.tools import loadData
from DeepLearn.tools import models


def main():
    batch_size = 128
    train_data, val_data, test_data = loadData.cifar10_data(batch_size)
    sample = next(iter(train_data))
    print(sample[0].shape, sample[1].shape)
    # # create model and train_data
    model = models.MyModel(32 * 32 * 3)
    model.compile(optimizer=optimizers.Adam(lr=1e-3),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=15, validation_data=val_data, validation_freq=2)

    model.evaluate(test_data)

    # model.save_weights('./models/cifar10')


if __name__ == '__main__':
    main()
