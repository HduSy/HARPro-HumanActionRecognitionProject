import numpy as np

a = np.arange(10)

np.random.shuffle(a)
print(a)
b = np.random.random((3, 10))
print(b)
c = np.random.randint(5, size=(10, 1))
print(c)
