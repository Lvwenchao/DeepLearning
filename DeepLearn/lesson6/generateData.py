# write by Mrlv
# coding:utf-8
import numpy as np
from matplotlib import pyplot
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

test_size = 0.3
x, y = make_moons(1000, noise=0.25, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
print('traindata:', x_train.shape, 'testdate:', y_test.shape)
