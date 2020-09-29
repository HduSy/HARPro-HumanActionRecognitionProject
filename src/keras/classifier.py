# minst手写体识别
# 数据预处理
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation  # 全连接层,激励函数
from keras.optimizers import RMSprop  # 优化器 加速神经网络训练

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(X_test.shape)  # (10000, 28, 28)
print(y_test.shape)  # (10000,)
print(y_test[:3])
# 数据预处理 像素/255 归一化为[0-1]
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
print(X_train.shape)  # (60000, 784)
X_test = X_test.reshape(X_test.shape[0], -1) / 255  # normalize
# 值转向量：将y数值转化为向量,y属于哪个值，哪个位置就为1,其余位为0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(y_test[:3])

# 建立神经网络
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# 定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

# 激活模型
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练网络的另一种方式
print('Training ------------')
model.fit(X_train, y_train, epochs=2, batch_size=32)  # 迭代训练两次每次处理一批32个

# 测试模型
print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

print('test before save: ', model.predict(X_test[0:2]))
model.save('my_model.h5')
del model
model = load_model('my_model.h5')
print('test after load: ', model.predict(X_test[0:2]))

"""
# save and load weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')
# save and load fresh network without trained weights
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
"""
