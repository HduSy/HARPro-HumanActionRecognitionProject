# from keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(y_train)
import numpy as np

arrar = np.array([1, 2, 3, 4, 5, 6])
print(np.split(arrar, 2))
print(type(arrar[:3]))
print(arrar[:3], arrar[3:])

feature = [
    [30.676, 21.378, 21.213, 10.0, 13.0, 23.409, 12.042, 13.038, 0.0, 1.0, 17.117, 29.0, 3.0, 16.155, 33.061, 31.623,
     64.498, 31.048, 64.498, 37.855, 36.878, 36.0, 35.609, 35.384, 29.0],
    [39.812, 28.018, 28.284, 14.866, 12.53, 28.071, 9.487, 13.601, 0.0, 3.0, 21.587, 43.105, 5.0, 22.561, 46.174,
     91.608, 40.311, 91.608, 38.013, 50.488, 50.99, 48.703, 47.539, 46.573, 45.044],
    [34.0, 31.016, 31.78, 41.11, 51.478, 32.28, 39.217, 50.99, 0.0, 6.0, 22.136, 36.878, 5.0, 21.19, 38.833, 36.056,
     36.056, 37.121, 37.336, 43.738, 43.174, 39.459, 41.976, 41.485, 37.483]]
label = ['running', 'boxing', 'handwaving']


def func(feature, label):
    return ((feature[:2], label[:2]), (feature[2:], label[2:]))


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = func(feature, label)
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)
