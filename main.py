import numpy as np

from _math import sigmoid
from _test import test1, test2, test3

def backpropagation(x, Y, w):
    for i in range(20000):
        current_Y = sigmoid(np.dot(x, w))
        error = Y - current_Y
        adjustment = np.dot(x.T, error * (current_Y * (1 - current_Y)))
        w += adjustment
    return w

train_x = np.array([
    [0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0]])
train_Y = np.array([[0, 1, 0, 0, 1]]).T

train_x2 = np.array([
    [0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]])
train_Y2 = np.array([[1, 0, 0, 1, 1]]).T
weights = 2 * np.random.random((7, 1)) - 1

print('Тренировочный набор входных данных')
print(train_x)
print('Тренировочный набор ожидаемых выходных данных')
print(train_Y)

weights = backpropagation(train_x, train_Y, weights)
weights = backpropagation(train_x2, train_Y2, weights)
print('Предсказание на тренировочном наборе данных')
print(sigmoid(np.dot(train_x, weights)))
test1(weights)
test2(weights)
test3(weights)