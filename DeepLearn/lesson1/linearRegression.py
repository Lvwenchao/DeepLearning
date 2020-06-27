# write by Mrlv
# coding:utf-8

# %%
import numpy as np


def compute_loss(w, b, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (w * x + b - y) ** 2
    totalError = totalError / float(len(points))
    return totalError


def update_w_b(w_starting, b_starting, lr, points):
    w_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w_gradient += (2 / N) * x * ((w_starting * x + b_starting) - y)
        b_gradient += (2 / N) * ((w_starting * x + b_starting) - y)
    new_w = w_starting - (lr * w_gradient)
    new_b = b_starting - (lr * b_gradient)
    return [new_w, new_b]


def gradient_runner(iterNum, points, w_start, b_start, lr):
    w = w_start
    b = b_start
    for i in range(iterNum):
        w, b = update_w_b(w, b, lr, np.array(points))
    return [w, b]


def run():
    points = np.loadtxt('../resource/data.csv', delimiter=',')
    w_start = 0
    b_start = 0
    iterationNum = 1000
    learning_rate = 0.0001
    print("gradient decent at w={0},b={1},loss={2}\n"
          .format(w_start, b_start, compute_loss(w_start, b_start, points)))
    print("running....\n")
    [w, b] = gradient_runner(iterationNum, points, w_start, b_start, learning_rate)
    print("Result:w={0},b={1},loss={2}\n"
          .format(w, b, compute_loss(w, b, points)))


if __name__ == "__main__":
    run()
