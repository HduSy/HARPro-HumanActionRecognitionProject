import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

model = Sequential()

model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
# 配置训练模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
config = model.get_config()
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))
one_hot_labels = np_utils.to_categorical(labels, num_classes=10)
# 迭代训练
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
