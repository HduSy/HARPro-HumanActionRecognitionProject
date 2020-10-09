from keras.engine.topology import Layer
from keras import backend as K


# 自定义注意力层
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    # 定义权重
    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        # super([Layer], self).build()
        super(AttentionLayer, self).build(input_shape)

    # 编写功能逻辑 inputs:输入张量
    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    # 编写修改输入张量形状逻辑
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
