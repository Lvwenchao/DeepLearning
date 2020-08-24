# write by Mrlv
# coding:utf-8
import os
import tensorflow as tf
from DeepLearn.tools.customize import CustomizeModel
from tensorflow.keras import datasets, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


# readData
batch_size = 128
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print('dataset:', x_train.shape, y_train.shape)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = train_db.map(preprocess).shuffle(60000).batch(batch_size)
test_db = test_db.map(preprocess).batch(batch_size)

db_iter = iter(train_db)
sample = next(db_iter)
print(sample[0].shape, sample[1].shape)

# create model
model = CustomizeModel(28 * 28)
model.build(input_shape=[None, 28 * 28])
model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.01),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_db, epochs=10, validation_data=test_db, validation_freq=2)

# 另一种方式
model.evaluate(test_db)
# predict
# %%
test_img = sample[0]
test_label = sample[1]
pre = model.predict(test_img)
test_label = tf.argmax(test_label, axis=1)
pre = tf.argmax(pre, axis=1)
print(pre)
print(test_label)

# save model
# %%
model.save_weights('./models/my_model')