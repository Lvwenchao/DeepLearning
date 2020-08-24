# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import datasets, optimizers
from DeepLearn.tools.preprocess import preprocess
from DeepLearn.tools import customize

batch_size = 128
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_val = tf.split(x_train, num_or_size_splits=[50000, 10000])
y_train, y_val = tf.split(y_train, num_or_size_splits=[50000, 10000])
print("data：", x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_train.min(), y_train.max())
y_train = tf.squeeze(y_train)
y_test = tf.squeeze(y_test)
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

train_date = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_date = train_date.map(preprocess).shuffle(10000).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.map(preprocess).batch(batch_size)

sample = next(iter(train_date))
print(sample[0].shape, sample[1].shape)

# create model and train_data
model = customize.MyModel()
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_date, epochs=15, validation_data=test_data, validation_freq=2)

# %%
model.save_weights('./models/cifar10')
