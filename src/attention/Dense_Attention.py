from keras.models import *
from keras.layers import Input, Dense, Multiply
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y

def build_model():
    K.clear_session() #清除之前的模型，省得压满内存
    inputs = Input(shape=(input_dim,)) #输入层

    # ATTENTION PART STARTS HERE 注意力层
    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
    attention_mul =  Multiply()([inputs, attention_probs])
    # ATTENTION PART FINISHES HERE

    attention_mul = Dense(64)(attention_mul) #原始的全连接
    output = Dense(1, activation='sigmoid')(attention_mul) #输出层
    model = Model(inputs=[inputs], outputs=output)
    return model


if __name__ == '__main__':
    np.random.seed(1337)  # for reproducibility
    input_dim = 32 #特征数
    N = 10000 #数据集总记录数
    inputs_1, outputs = get_data(N, input_dim) #构造数据集

    m = build_model() #构造模型
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()

    m.fit([inputs_1], outputs, epochs=20, batch_size=64, validation_split=0.2)

    testing_inputs_1, testing_outputs = get_data(1, input_dim)

    # Attention vector corresponds to the second matrix.
    # The first one is the Inputs output.
    attention_vector = get_activations(m, testing_inputs_1,
                                       print_shape_only=True,
                                       layer_name='attention_vec')[0].flatten()
    print('attention =', attention_vector)

    # plot part.


    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')
    plt.show()
