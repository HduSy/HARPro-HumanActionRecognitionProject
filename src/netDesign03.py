import shutil
from glob import glob
from time import time
import keras
from keras.layers import LSTM, GRU
from keras.layers import Dense, Activation, Dropout, LeakyReLU, Input
from keras.layers import add, subtract, multiply, average, maximum, concatenate, dot
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from src.generateMnistImitationData import makeDataSetFromTxt, spliceDataSet
from src.keras.selflayers.AttentionLayer import AttentionLayer
import os
import keras.backend as K
from src.public import txtDir, actions, model_filename, spatial_attention, temporal_attention
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


# reduce_lr = LearningRateScheduler(scheduler)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

os.environ["PATH"] += os.pathsep + 'D:\\Program Files (x86)\\Graphviz2.38\\bin'  # 注意修改你的路径
# model = Sequential()
model = None
learning_rate = 0.001
training_iters = 27  # 27
dropout_rate = 0.5
# TODO:使用DropOut解决过拟合问题
# training_iters = 5
batch_size = 128  # 128  # 每次梯度更新的样本数
# TODO:使用注意力机制前后对模型准确率的影响
# https://zhuanlan.zhihu.com/p/77609689


n_step = 25
n_input = 25
n_hidden = 128  # 128
n_classes = 7  # 动作类别数

from attention import Attention


# 模型训练
def train(x1_train, x2_train, y1_train, x1_vali, x2_vali, y1_vali):
    print('begin model training...')
    global model
    adam = Adam(lr=learning_rate)
    checkpoint_dir = 'checkpoints'
    checkpoints = glob(os.path.join(checkpoint_dir, '*.h5'))
    # 定义两个分支
    inputA = Input(shape=(n_step, n_input))
    inputB = Input(shape=(n_step, n_input))
    # 空间注意力模块-空间关键点选择门
    # return_sequences:true返回所有中间隐藏值false返回最后一个隐藏值
    # shape=(samples, time_steps, input_dim)
    if spatial_attention:
        print('空间注意力机制')
        x1 = GRU(n_hidden, batch_input_shape=(1, n_step, n_input), return_sequences=True,
                 unroll=True)(inputA)
        x1 = AttentionLayer()(x1)
        # x1 = Attention()(x1)
    else:
        print('未使用空间注意力机制')
        x1 = GRU(n_hidden, batch_input_shape=(1, n_step, n_input), return_sequences=False,
                 unroll=True)(inputA)
    x1 = Dense(24, activation='tanh')(x1)
    spatialModal = Model(inputs=inputA, outputs=x1)
    # 时间注意力模块-时间关键帧选择门
    if temporal_attention:
        print('时间注意力机制')
        x2 = GRU(n_hidden, batch_input_shape=(1, n_step, n_input), return_sequences=True,
                 unroll=True)(inputB)
        x2 = AttentionLayer()(x2)
    else:
        print('未使用时间注意力机制')
        x2 = GRU(n_hidden, batch_input_shape=(1, n_step, n_input), return_sequences=False,
                 unroll=True)(inputB)
    x2 = Dense(24, activation='relu')(x2)
    temporalModal = Model(inputs=inputB, outputs=x2)

    # 特征做融合-后面主GRU可能就不需要再加注意力模型了
    # combined = add([spatialModal.output, temporalModal.output])
    # combined = average([spatialModal.output, temporalModal.output])
    # combined = maximum([spatialModal.output, temporalModal.output])
    combined = concatenate([spatialModal.output, temporalModal.output])
    # combined = dot([spatialModal.output, spatialModal.output])
    # z = GRU(n_hidden, batch_input_shape=(None, n_step, n_input), return_sequences=False, unroll=True)(combined)
    z = Dense(24)(combined)
    z = Dropout(dropout_rate)(z)
    z = Dense(n_classes, activation='softmax')(z)

    model = Model(inputs=[spatialModal.input, temporalModal.input], outputs=z)
    model.summary()  # 输出模型各层的参数状况
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # delete folder and its content and creates a new one.
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    # 保存验证集上表现最好的某次模型
    checkpoint = ModelCheckpoint(monitor='val_accuracy', save_best_only=True,
                                 filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_accuracy:.3f}.h5'))
    # validation_data=([x1_vali, x2_vali], y1_vali), callbacks=[reduce_lr],
    hist = model.fit([x1_train, x2_train], y1_train, validation_split=0.33,
                     validation_data=([x1_vali, x2_vali], y1_vali),callbacks=[checkpoint],
                     batch_size=batch_size, epochs=training_iters, verbose=1)
    print(hist.history)
    print('model trained.')


# 模型测试
def test():
    print('begin model testing...')
    global model
    scores = model.evaluate([spatial_vali, temporal_vali], y_vali, batch_size=32, verbose=1)
    # loss, accuracy
    print('GRU test score:', scores[0])
    print('GRU test accuracy:', scores[1])
    print('model tested.')


def saveModelPng():
    global model
    plot_model(model, to_file='model.png', show_shapes=True)


# 模型保存
def save():
    print('begin model saving...')
    global model
    model.save(model_filename)
    del model
    print('model saved.')


if __name__ == '__main__':
    begin = time()
    ((spatial_train, temporal_train, y_train), (spatial_vali, temporal_vali, y_vali)) = spliceDataSet(test=False)
    end = time()
    print('程序处理时长约%.1fmin' % ((end - begin) / 60))
    print(spatial_train.shape)  # (7057, 25, 25)
    print(temporal_train.shape)  # (7057, 25, 25)
    print(y_train.shape)  # (7057, )
    for i in range(len(y_train)):
        y_train[i] = actions.index(y_train[i])
    for i in range(len(y_vali)):
        y_vali[i] = actions.index(y_vali[i])
    # print(y_train[0], y_train[1], y_train[2])
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_vali = keras.utils.to_categorical(y_vali, n_classes)
    # print(x_train[0])
    # print(y_train[0], y_train[1], y_train[2])
    begin = time()
    train(spatial_train, temporal_train, y_train, spatial_vali, temporal_vali, y_vali)
    end = time()
    print('程序训练时长约%.1fmin' % ((end - begin) / 60))
    test()
    saveModelPng()
    save()