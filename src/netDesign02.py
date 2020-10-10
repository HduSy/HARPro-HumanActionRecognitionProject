from time import time
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from src.generateMnistImitationData import readDataFromTxt
from src.keras.selflayers.AttentionLayer import AttentionLayer

model = Sequential()
learning_rate = 0.001
training_iters = 30
# training_iters = 5
batch_size = 128
display_step = 10
add_attention = True  # False
# TODO:使用注意力机制前后对模型准确率的影响
# without attention
# LSTM test score: 0.2682137036779124
# LSTM test accuracy: 0.8789808750152588
# LSTM test score: 0.19444831573659446
# LSTM test accuracy: 0.9197452068328857
# with attention
# LSTM test score: 0.1839267789937888
# LSTM test accuracy: 0.9299362897872925
# LSTM test score: 0.1824791385489664
# LSTM test accuracy: 0.9363057613372803
dropout_rate = 0.8
# TODO:使用DropOut解决过拟合问题
# with dropout layer
# 0.9
# LSTM test score: 0.16199284065870723
# LSTM test accuracy: 0.9299362897872925
# 0.8 the best the best the best the best
# LSTM test score: 0.10659453525285052
# LSTM test accuracy: 0.9617834687232971
# LSTM test score: 0.12854519761671687
# LSTM test accuracy: 0.9426751732826233
# LSTM test score: 0.1091721070135475
# LSTM test accuracy: 0.9554139971733093
# 0.7
# LSTM test score: 0.10670858781049206
# LSTM test accuracy: 0.9605095386505127
# 0.6
# LSTM test score: 0.1138901218487199
# LSTM test accuracy: 0.9579617977142334
# 0.5
# LSTM test score: 0.1579790642971446
# LSTM test accuracy: 0.946496844291687
# 0.4
# LSTM test score: 0.10913121895805286
# LSTM test accuracy: 0.9579617977142334
# 0.3
# LSTM test score: 0.11853743914016493
# LSTM test accuracy: 0.9554139971733093
# 0.2
# LSTM test score: 0.15626057428159532
# LSTM test accuracy: 0.950318455696106

n_step = 25
n_input = 25
n_hidden = 128
n_classes = 6

txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action'
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


def train(x_train, y_train, x_test, y_test):
    print('begin training...')
    global model
    adam = Adam(lr=learning_rate)

    if add_attention:
        print('使用注意力机制')
        model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), return_sequences=True, unroll=True))
        model.add(AttentionLayer())
    else:
        print('未使用注意力机制')
        model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), return_sequences=False, unroll=True))
    model.add(Dropout(dropout_rate))
    # model.add(Dense(2, activation='sigmoid'))  # 二分类问题sigmoid等价于softmax 分为静态和动态动作
    model.add(Dense(n_classes, activation='softmax'))

    model.summary()  # 输出模型各层的参数状况
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=training_iters, verbose=1,
              validation_data=(x_test, y_test))
    print('training end...')


def test():
    print('start testing...')
    global model
    scores = model.evaluate(x_test, y_test, verbose=0)
    # TODO:scores长什么样，分别代表什么意义
    print('LSTM test score:', scores[0])
    print('LSTM test accuracy:', scores[1])
    print('testing end...')


if __name__ == '__main__':
    begin = time()
    ((x_train, y_train), (x_test, y_test)) = readDataFromTxt(txtDir)
    end = time()
    print('程序处理时长约%.1fmin' % ((end - begin) / 60))
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    for i in range(len(y_train)):
        y_train[i] = actions.index(y_train[i])
    for i in range(len(y_test)):
        y_test[i] = actions.index(y_test[i])
    # print(y_train[0], y_train[1], y_train[2])
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    # print(x_train[0])
    # print(y_train[0], y_train[1], y_train[2])
    begin = time()
    train(x_train, y_train, x_test, y_test)
    end = time()
    test()
    print('程序训练时长约%.1fmin' % ((end - begin) / 60))
