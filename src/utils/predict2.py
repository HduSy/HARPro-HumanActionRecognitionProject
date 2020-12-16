from keras.models import load_model

from src.generateOneActionImitationData import readOneActionDataFromTxt
from src.keras.selflayers.AttentionLayer import AttentionLayer
from src.generateMnistImitationData import makeDataSetFromTxt, spliceDataSet
from src.public import spatial_attention, temporal_attention
import numpy as np

from src.readTxtData2Memory import transformTxtLine2ListObj
from src.public import actions, txtDir, model_filename

action_type = 'falling'
action_sum = 0
action_right = 0
if spatial_attention or temporal_attention:
    model = load_model(model_filename, custom_objects={'AttentionLayer': AttentionLayer})
else:
    model = load_model(model_filename)


# 样本,待预测动作,样本动作
# sample, action_type, y_test[i]
def predict(sample, target_label, sample_label):
    # 25*25   25*25   'action label'
    # 'spatial_feature:', sample[0], 'temporal_feature:', sample[1],
    # print('待预测动作', target_label, '样本动作', sample_label)
    # input()
    global action_sum, action_right
    # Expected to see 2 array(s)
    # Error when checking input: expected input_1 to have 3 dimensions, but got array with shape (25, 25)
    y = model.predict(x=sample, batch_size=1, verbose=1)
    y_index = np.where(y == y.max())[1][0]
    # 若样本动作==待预测动作，该动作总样本数+1
    if target_label == sample_label:
        action_sum += 1
        # 再若预测值==样本值，该动作预测正确数+1
        if actions[y_index] == sample_label:
            action_right += 1
            print('预测正确')
        # 把待预测动作预测为了actions[y_index]动作
        else:
            print('预测动作:', actions[y_index], '不是', '样本动作:', sample_label)
    else:
        print('样本动作不是待预测动作')
    # print('待预测动作:', target_label, '样本动作:', sample_label, '预测动作:', actions[y_index])
    # 若待预测动作===样本动作&预测值==样本值，该动作预测正确数+1
    # if target_label == sample_label and actions[y_index] == sample_label:
    #     action_right += 1
    # elif target_label == sample_label and actions[y_index] != sample_label:
    #     print('待预测动作:', target_label, '样本动作:', sample_label, '预测动作:', actions[y_index])
    # return actions[y_index] == sample_label
    # return (action_right / action_sum) * 100


if __name__ == '__main__':
    # txtDir += '\\' + action_type
    # ((x_train, y_train), (x_test, y_test)) = readOneActionDataFromTxt(txtDir)
    # todo:若将某一动作训练验证分开的话，这种调用是可行的
    # ((spatial_train, temporal_train, y_train), (spatial_test, temporal_test, y_test)) = readOneActionDataFromTxt(txtDir, action_type)
    # (spatial_test, temporal_test, y_test) = makeDataSetFromTxt(txtDir, test=True)
    (spatial_test, temporal_test, y_test) = spliceDataSet(test=True)
    # step1.读取某种动作数据(训练与测试分开)
    # (spatial_test, temporal_test, y_test) = makeDataSetFromTxt(txtDir, True)
    right = 0
    print(spatial_test.shape)
    sample_num = spatial_test.shape[0]
    # step2.模型预测结果与该序列对应label是否相等计算准确率
    # for i in range(sample_num):
    #     if predict([np.array(spatial_test[i]).reshape(1, 25, 25), np.array(temporal_test[i]).reshape(1, 25, 25)],
    #                action_type, y_test[i]):
    #         right += 1
    for i in range(sample_num):
        predict([np.array(spatial_test[i]).reshape(1, 25, 25), np.array(temporal_test[i]).reshape(1, 25, 25)],
                action_type, y_test[i])
    print(action_right, action_sum)
    accuracy = float(action_right / action_sum) * 100
    # accuracy = predict([np.array(spatial_test), np.array(temporal_test)], action_type)
    print('{0}预测准确率:{1}'.format(action_type, accuracy))
    pass
