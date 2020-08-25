# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, optimizers
from DeepLearn.tools.customize import RNN
import os

# load_dat
batchsz = 128
total_words = 10000
embedding_len = 100
max_words_len = 80
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_words_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_words_len)
print('x_train.shape:', x_train.shape, 'y_shape', y_train.shape, 'y_max：', max(y_train), 'x_min：', min(y_train))

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(1000).batch(batchsz, drop_remainder=True)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.shuffle(1000).batch(batchsz, drop_remainder=True)

simple = next(iter(train_data))
print(simple[0].shape, simple[1].shape)

# file save path
checkpoint_path = r'E:\pythonProject\DeepLearn\resources\models\rnn\rnn.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 verbose=1)


def main():
    units = 64
    epochs = 10
    model = RNN(units, embedding_len, max_words_len, total_words)

    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                  )

    model.fit(train_data,
              epochs=epochs,
              validation_data=test_data,
              callbacks=[cp_callback])

    model.evaluate(test_data)


def test_model():
    units = 64
    model = RNN(units, embedding_len, max_words_len, total_words)
    model.load_weights(checkpoint_path)

    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.evaluate(test_data)


if __name__ == '__main__':
    test_model()
