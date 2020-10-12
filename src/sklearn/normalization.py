from sklearn.preprocessing import MinMaxScaler
import numpy as np

min_max_scaler = MinMaxScaler(feature_range=(0, 1))

# test2 = np.array([[[1, 1], [2, 2]]])
# test2_minmax = min_max_scaler.fit_transform(test2)
# print(test2_minmax)  # Found array with dim 3. MinMaxScaler expected <= 2

x = np.array([[1, 1], [2, 2], [3, 3]])
x_minmax = min_max_scaler.fit_transform(x)
print(x_minmax)

test = np.array([[10, 10], [20, 20]])
test_minmax = min_max_scaler.fit_transform(test)
print(test_minmax)

# normalized successfully!
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# 讲道理对于这样的一维张量，这种是一种达到目的正则化简单做法。对于(x,y) 25*25 数据，应对x,y坐标两个维度单独分别正则化
frame01 = np.array([22.361, 16.031, 15.524, 10.63, 9.0, 16.763, 12.806, 13.0, 0.0, 2.0, 12.369, 25.0, 3.0, 11.045,
                    21.84, 23.195, 23.537, 23.0, 88.193, 23.195, 23.195, 23.087, 25.632, 26.571, 26.926]).reshape(-1, 1)
test_frame_sF = min_max_scaler.fit_transform(frame01)
print(test_frame_sF.reshape(1, -1))

frame02 = np.array([-21.185, 36.414, 19.964, 4.986, -26.635, -17.558, 14.514, 23.592, 45.,
                    30.419, 13.572, -1.009, -41.15, 31.428, 5.45, 6.322, -9.077, 14.581,
                    11.274, 12.445, -41.199, -14.978, -14.036, -9.622, 10.67]).reshape(-1, 1)
test_frame_tf = min_max_scaler.fit_transform(frame02)
print(test_frame_tf.reshape(1, -1))

test_negtive = np.array([-21.185, 36.414, 19.964, 4.986, -26.635, -17.558, 14.514, 23.592, 45.,
                         30.419, 13.572, -1.009, -41.15, 31.428, 5.45, 6.322, -9.077, 14.581,
                         11.274, 12.445, -41.199,
                         -14.978, -14.036, -9.622, 10.67]).reshape(-1, 1)
test_negtive_minmax = min_max_scaler.fit_transform(test_negtive)
print(test_negtive_minmax.reshape(1, -1))

# 归一化[-1, 1]
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
test_n2a = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]).reshape(-1, 1)  # (9, 1)

test_n2a_minmax = min_max_scaler.fit_transform(test_n2a).reshape(1, -1)  # (1, 9)
print(test_n2a_minmax)
