from time import time
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

from src.generateMnistImitationData import readDataFromTxt

model = Sequential()
learning_rate = 0.001
training_iters = 20
# training_iters = 5
batch_size = 128
display_step = 10

n_input = 25
n_step = 25
n_hidden = 128
n_classes = 6

txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action'
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


def train(x_train, y_train, x_test, y_test):
    print('begin training...')
    global model
    model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), unroll=True))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    adam = Adam(lr=learning_rate)
    model.summary()
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=training_iters, verbose=1,
              validation_data=(x_test, y_test))
    print('training end...')


def test():
    print('start testing...')
    global model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('LSTM test score:', scores[0])
    print('LSTM test accuracy:', scores[1])
    print('testing end...')


if __name__ == '__main__':
    start = time()
    ((x_train, y_train), (x_test, y_test)) = readDataFromTxt(txtDir)
    stop = time()
    print('程序处理时长%fs' % (stop - start))
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
