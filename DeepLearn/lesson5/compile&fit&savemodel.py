# write by Mrlv
# coding:utf-8
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from DeepLearn.tools.loadData import minist_data
from DeepLearn.tools import models
from tensorflow.keras import datasets, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_image(predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label, img
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap='gray')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(predictions_array),
                                         true_label,
                                         color=color))


def main():
    # loadData
    batch_size = 128
    train_data, test_data = minist_data(batch_size)
    sample = next(iter(train_data))
    print(sample[0].shape, sample[1].shape)
    # train
    model = models.MyModel(28 * 28)
    model.build([None, 28 * 28])
    model.summary()

    model.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=20, validation_data=test_data, validation_freq=5)  # validation_freq 表示每两次计算一次平均值

    model.evaluate(test_data)

    model.save_weights(r'E:\pythonProject\DeepLearning\resources\models\mnist', save_format='tf')
    # predict


if __name__ == '__main__':
    main()
