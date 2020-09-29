import numpy as np

np.random.seed(1337)
from keras.models import Sequential  # 一层一层建立神经层
from keras.layers import Dense  # 全连接层
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)  # 随机打乱数据
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # 训练数据
X_test, Y_test = X[160:], Y[160:]  # 测试数据

# 新建模型
model = Sequential()
# 添加神经层
model.add(Dense(output_dim=1, input_dim=1))
# 激活神经网络 损失函数：均方误差 优化器：随机梯度下降法
model.compile(loss='mse', optimizer='sgd')
# 训练模型
print('Training......')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)  # 一批一批训练
    if step % 100 == 0:
        print('train cost:', cost)
# 检验模型
print('\nTesting......')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 可视化结果
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
