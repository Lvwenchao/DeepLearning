# write by Mrlv
# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def himmelblau(points):
    return (points[0] ** 2 + points[1] - 11) ** 2 + (points[0] + points[1] ** 2 - 7) ** 2


def main():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    print(X.shape, Y.shape)
    Z = himmelblau([X, Y])

    fig = plt.figure('himmelblau')
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    startpoints = tf.constant([[4., 0.], [1., 0.], [-4., 0.], [-2., 2.]])
    endpoints = []
    for point in startpoints:
        for epoch in range(100):
            with tf.GradientTape() as tape:
                tape.watch(point)
                y = himmelblau(point)

            grads = tape.gradient(y, point)[0]
            point += (-0.01) * grads
        endpoints.append(point)

    print(endpoints)


if __name__ == '__main__':
    main()
