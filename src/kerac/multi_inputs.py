import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Input
from tensorflow.keras.models import Model

import keract
import matplotlib.pyplot as plt
import pandas as pd
from src.kerac.utils import print_names_and_values, print_names_and_shapes, gpu_dynamic_mem_growth

# gradients requires no eager execution.
tf.compat.v1.disable_eager_execution()


def get_multi_inputs_model():
    a = Input(shape=(10,))
    b = Input(shape=(10,))
    c = Add()([a, b])
    c = Dense(1, activation='sigmoid', name='last_layer')(c)
    m_multi = Model(inputs=[a, b], outputs=c)
    return m_multi


def get_single_inputs_model():
    inputs = Input(shape=(10,))
    x = Dense(1, activation='sigmoid')(inputs)
    m_single = Model(inputs=[inputs], outputs=x)
    return m_single


def main():
    np.random.seed(123)
    inp_a = np.random.uniform(size=(5, 10))
    inp_b = np.random.uniform(size=(5, 10))
    out_c = np.random.uniform(size=(5, 1))

    # Just for visual purposes.
    np.set_printoptions(precision=2)

    # Activations of all the layers
    print('MULTI-INPUT MODEL')
    m1 = get_multi_inputs_model()
    m1.compile(optimizer='adam', loss='mse')
    print_names_and_values(keract.get_activations(m1, [inp_a, inp_b]))
    print_names_and_values(keract.get_gradients_of_trainable_weights(m1, [inp_a, inp_b], out_c))
    print_names_and_values(keract.get_gradients_of_activations(m1, [inp_a, inp_b], out_c))

    # Just get the last layer!
    print_names_and_values(keract.get_activations(m1, [inp_a, inp_b], layer_names='last_layer'))
    print_names_and_values(keract.get_gradients_of_activations(m1, [inp_a, inp_b], out_c,
                                                               layer_names='last_layer'))
    print('')

    print('SINGLE-INPUT MODEL')
    m2 = get_single_inputs_model()
    m2.compile(optimizer='adam', loss='mse')
    print_names_and_values(keract.get_activations(m2, inp_a))
    print_names_and_values(keract.get_gradients_of_trainable_weights(m2, inp_a, out_c))
    print_names_and_values(keract.get_gradients_of_activations(m2, inp_a, out_c))


if __name__ == '__main__':
    main()
