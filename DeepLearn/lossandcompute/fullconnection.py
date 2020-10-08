# write by Mrlv
# coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import pandas as pd

a = tf.random.uniform([4, 256], maxval=50)
model = Sequential([layers.Dense(128, activation='relu'),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(10)])
model.build([None, 256])
model.summary()
for p in model.variables:
    print(p.name, p.shape)
