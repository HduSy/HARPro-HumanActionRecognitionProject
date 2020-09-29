import numpy as np

# https://blog.csdn.net/weixin_40679412/article/details/82949079
# 计算逆矩阵
a = np.mat([[1, 2, 3], [2, 5, 4], [1, 2, 2]])
inva = a.I
# inva = np.linalg.inv(a)
print(a * inva)
# 求解线性方程组 numpy.linalg中的函数solve可以求解形如 Ax = b 的线性方程组，其中 A 为矩阵，b 为一维或二维的数组
# x 是未知变量，x = np.linalg.solve(A, b)，就可以这么去求解未知变量x
A = np.mat([[1, -2, 1], [0, 2, -8], [-4, 5, 9]])
b = np.array([0, 8, 9])
x = np.linalg.solve(A, b)
print(x)

print(np.dot(A, x))  # [0, 8, 9]
