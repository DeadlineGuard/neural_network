import numpy as np

from .math import sigmoid
from .test import 

def backpropagation(x, Y, w):
    current_Y = None

    for i in range(10000):
        current_Y = sigmoid(np.dot(x, w))
        error = Y - current_Y
        adjustment = np.dot(train_X.T, error * (current_Y * (1 - current_Y)))
        w += adjustment

    return current_Y

train_x = np.array([
    [0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0]])
train_Y = np.array([[0, 1, 0, 0, 1]]).T
weights = 2 * np.random.random((7,1)) - 1

Y = backpropagation(train_x, train_Y, weights)
print(Y)