from keract import keract
from keras import Input, Model
from keras.layers import Add, Dense, Dropout, GRU, concatenate
from keras.optimizers import Adam

from src.generateMnistImitationData import spliceDataSet
from src.kerac.utils import print_names_and_values
from src.keras.selflayers.AttentionLayer import AttentionLayer
from src.netDesign03 import n_hidden, n_step, n_input, dropout_rate, n_classes, learning_rate
from src.public import spatial_attention, temporal_attention

import pandas as pd
import matplotlib.pyplot as plt

model = None


def get_multi_inputs_model():
    global model
    inputA = Input(shape=(n_step, n_input))
    inputB = Input(shape=(n_step, n_input))
    # 空间注意力模块-空间关键点选择门
    if spatial_attention:
        print('空间注意力机制')
        x1 = GRU(n_hidden, batch_input_shape=(1, n_step, n_input), return_sequences=True,
                 unroll=True)(inputA)
        x1 = AttentionLayer()(x1)
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
    combined = concatenate([spatialModal.output, temporalModal.output])
    z = Dense(24)(combined)
    z = Dropout(dropout_rate)(z)
    z = Dense(n_classes, activation='softmax')(z)
    model = Model(inputs=[spatialModal.input, temporalModal.input], outputs=z)
    return model


from src.attention.Dense_Attention import get_activations

if __name__ == '__main__':
    m1 = get_multi_inputs_model()
    m1.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    ((inp_a, inp_b, out_c), (inp_d, inp_e, out_f)) = spliceDataSet(test=False)
    # print_names_and_values(get_activations(m1, [inp_a, inp_b]))
    # print_names_and_values(keract.get_gradients_of_trainable_weights(m1, [inp_a, inp_b], out_c))
    # print_names_and_values(keract.get_gradients_of_activations(m1, [inp_a, inp_b], out_c))
    # Just get the last layer!
    # print_names_and_values(keract.get_activations(m1, [inp_a, inp_b], layer_names='last_layer'))
    # print_names_and_values(keract.get_gradients_of_activations(m1, [inp_a, inp_b], out_c, layer_names='last_layer'))
    print(inp_d.shape)  # (1021, 25, 25)
    attention_vector = get_activations(m1, [inp_d, inp_e],
                                       print_shape_only=True,
                                       layer_name='attention_layer_1')[0].flatten()
    print('attention =', attention_vector)
    # plot part.
    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')
    plt.show()
