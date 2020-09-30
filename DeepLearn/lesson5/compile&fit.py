# write by Mrlv
# coding:utf-8
import os
import tensorflow as tf
from DeepLearn.tools.loadData import minist_data
from DeepLearn.tools.models import CustomizeModel
from tensorflow.keras import datasets, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # loadData
    batch_size = 128
    train_data, test_data = minist_data(batch_size)
    sample = next(iter(train_data))
    print(sample[0].shape, sample[1].shape)

    # train
    model = CustomizeModel(28 * 28)
    model.build([None, 28 * 28])
    model.summary()

    model.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=10, validation_data=test_data, validation_freq=2)  # validation_freq 表示每两次计算一次平均值

    # predict
    test_img = sample[0]
    test_label = sample[1]
    truth = tf.argmax(test_label, axis=1)
    pre = model.predict(test_img)
    pre = tf.argmax(pre, axis=1)
    print(tf.cast(tf.equal(pre, truth), dtype=tf.int8))
    print(pre)
    print(truth)



if __name__ == '__main__':
    main()
