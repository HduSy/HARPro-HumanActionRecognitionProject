from time import time
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation
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
# with attention
# LSTM test score: 0.1839267789937888
# LSTM test accuracy: 0.9299362897872925

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
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

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
    train(x_train, y_train, x_test, y_test)
    test()
