import numpy as np

from _math import sigmoid

def predict(x, Y, w):
    print('Тестовый набор входных данных')
    print(x)
    print('Тестовый набор ожидаемых выходных данных')
    print(Y)
    print('Предсказанные значения')
    print(sigmoid(np.dot(x, w)))

def test1(w):
    x = np.array([
    [0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]])
    Y = np.array([[1, 0, 0, 1, 1]]).T # Пятый столбик
    predict(x, Y, w)

def test2(w):
    x = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0]])
    Y = np.array([[0, 1, 1, 0, 0]]).T # И снова пятый столбик
    predict(x, Y, w)
