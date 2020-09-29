import numpy as np

# -1：元素总个数除以已固定的维数
a = np.arange(10).reshape((2, -1))  # (2,5)
b = np.arange(10).reshape((5, -1))  # (5,2)
c = np.arange(10).reshape(-1, 2)  # (5,2)

print(a)
print(b)
print(c)
print(a.shape)  # (2,5) 元组

# None 增加一维，数据量不变 跟keras里意义好像不同
# d = np.ones((2, 4, 2, 3))
# print('d.shape:', d.shape)
# d0 = d[0, :, :, :]  # 缺失一维
# print('d0.shape:', d0.shape)
# d1 = d[None, :, :, :, :]
# print('d1.shape:', d1.shape)
# d2 = d[:, :, :, :, None]
# print('d2.shape:', d2.shape)
