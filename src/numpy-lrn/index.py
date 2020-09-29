import numpy as np
import matplotlib.pyplot as plt

zeroArr = np.zeros((2, 2))
oneArr = np.ones((2, 3))
# print(zeroArr.astype(int))
# print(oneArr.astype(int))

a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
# b = np.array((1, 2, 3, 4, 5, 6))
# c = np.arange(5)
# d = np.linspace(0, 2 * np.pi, 5)
# print(a)
# print(b)
# print(c)
# print(d)
# 多维数组切片规则:对以逗号分隔的不同维度单独切片
print(a[0, 1:4])  # [12 13 14]
print(a[1:4, 0])  # [16 21 26]
print(type(a))
print(a.shape)
print(a.itemsize)
print(a.nbytes)
print(a.ndim)
print(a.dtype)

b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78])
b = b.reshape((5, 5))
print(b)

c = np.arange(25)
c = c.reshape((5, 5))
print(c)
# 除点乘运算符外逐元素操作
# print(b + c)
# print(b - c)
# print(b * c)
# print(c / b)
print(c ** 2)
print(b > c)
# print(b.dot(c)) # 矩阵相乘
d = np.arange(10)
print(d.min())
print(d.max())
print(d.sum())
print(d.cumsum())  # 累计和 0 1 3 6 10 ...
# 独特的索引方式
a = np.arange(0, 100, 10)
indices = [1, 5, -1]
b = a[indices]
print(b)
# 布尔屏蔽
a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a, b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
# plt.show()
# 缺省索引
a = np.arange(0, 100, 10)
b = a[:5]
c = a[a >= 50]
print(b)  # >>>[ 0 10 20 30 40]
print(c)  # >>>[50 60 70 80 90]
# 方法
a = np.zeros((2, 2))
b = np.ones((2, 2))
c = np.full((2, 2), 7)
d = np.eye(2, 2)
e = np.random.random((2, 2))
print(a)
print(b)
print(c)
print(d)
print(e)
# ndarray是多维的需要为每一维指定切片
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(a[:2, 1:3])  # [[2,3],[6,7]]
# 高级索引之整数数组索引
'''
整数索引允许使用另一个数组的数据构造新的数组
整数数组中每个整数代表取该维度第几项
'''
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 2], [1, 1]])  # [2,6]
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print('我们的数组是：')
print(x)
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols]
print('这个数组的四个角元素是：')
print(y)
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(a)
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])
a[np.arange(4), b] += 10
print(a)
# bool索引
a = np.array([[1, 2], [3, 4], [5, 6]])
b = a > 2
print(b)
print(a[b])

x = np.array([1, 2], dtype=int)
y = np.array([3, 4], dtype=int)
print(np.add(x, y))
print(np.subtract(x, y))
print(np.multiply(x, y))
print(np.divide(x, y))
print(np.sqrt(y * y))
print(np.dot(x, y))  # 矩阵相乘 向量内积 1*3+2*4
x = np.array([[1, 2], [3, 4]])
y = np.array([[1, 2], [3, 4]])
print(x.dot(y))  # [[ 7 10],[15 22]]
x = np.array([[1, 2], [3, 4]])
y = [[1, 2], [3, 4]]
print(np.dot(x, y))
print(x.dot(y))
# sum
print(x.sum())  # 10
print(x.sum(axis=0))  # 第一维上 [4,6]
print(x.sum(axis=1))  # 第二维上 [3,7]
# T

print(x.T)  # 转置 [[1,3],[2,4]]

y0 = [1, 2, 3, 4, 5, 6]
y = [y0[1:3]]
print('y',y)
