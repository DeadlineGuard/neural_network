import numpy as np

def predict(x, Y, w):
    print('Тестовый набор входных данных')
    print(x)
    print('Тестовый набор ожидаемых выходных данных')
    print(Y)
    print('Предсказанные значения')
    print(np.dot(x, w))

# Добавь данные в тесты
# Данные должны находиться в np.array()
# Нужно придумать входные и ожидаемые выходные данные
# Их нужно записать вместо None
# Способ записи данных есть в main.py у переменных train_x и train_Y
def test1(w):
    x = None
    Y = None
    predict(x, Y, w)

def test2(w):
    x = None
    Y = None
    predict(x, Y, w)