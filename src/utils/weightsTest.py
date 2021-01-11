from keras.models import load_model
import numpy as np

from src.keras.selflayers.AttentionLayer import AttentionLayer
from src.public import spatial_attention, temporal_attention
from src.public import model_filename

action_type = 'falling'
action_sum = 0
action_right = 0
if spatial_attention or temporal_attention:
    model = load_model(model_filename, custom_objects={'AttentionLayer': AttentionLayer})
else:
    model = load_model(model_filename)

spatial = model.get_layer('attention_layer_1').get_weights()
print(spatial[0].shape)
temporal = model.get_layer('attention_layer_2').get_weights()
print(temporal[0].shape)


def getSum():
    sum = 0
    for i in range(len(spatial)):
        sum += spatial[i]
    print(sum)  # should be 1
