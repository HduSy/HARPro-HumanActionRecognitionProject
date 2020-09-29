import numpy as np

v = np.array([1, 2, 3])
w = np.array([4, 5])
vv = np.reshape(v, (3, 1))

print(vv * w)  # [[4,5],[8,10],[12,15]]

x = np.array([[1, 2, 3], [4, 5, 6]])

print(x + v)  # [[2,4,6],[5,7,9]]
print(v.T)
print(np.transpose(v))
