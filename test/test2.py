import numpy as np

tp = 1, 2, 3, 4, 5
print(type(tp))
print(tp)
nptp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)

print(nptp.shape)
print(nptp.ndim)
print(nptp.itemsize)  # 1
print(nptp)
# 类型转换
nptp = nptp.astype(np.float64)
print(nptp)
print(nptp.itemsize)  # 8
(x, y), (x1, y1) = ((1, 2), (3, 4))
# print(x, y, x1, y1)
