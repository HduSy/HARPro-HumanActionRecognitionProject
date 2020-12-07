from keras.models import load_model

from src.generateOneActionImitationData import readOneActionDataFromTxt
from src.keras.selflayers.AttentionLayer import AttentionLayer
# from src.generateMnistImitationData import readDataFromTxt
import numpy as np

from src.readTxtData2Memory import transformTxtLine2ListObj

actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking', 'falling']
txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action'
action_type = 'boxing'
model_filename = 'F:\\XLDownload\\dataSet\\KTH\HARPro\\src\\model-file\\HAR.h5'

model = load_model(model_filename, custom_objects={'AttentionLayer': AttentionLayer})


def predict(sample, label):
    y = model.predict(x=sample, verbose=1)
    y_index = np.where(y == y.max())[1][0]
    print(actions[y_index])
    return actions[y_index] == label


if __name__ == '__main__':
    txtDir += '\\' + action_type
    # ((x_train, y_train), (x_test, y_test)) = readOneActionDataFromTxt(txtDir)
    ((spatial_train, temporal_train, y_train), (spatial_test, temporal_test, y_test)) = readOneActionDataFromTxt(txtDir, action_type)
    right = 0
    print(spatial_test.shape)
    sample_num = spatial_test.shape[0]
    # todo:有bug
    for i in range(sample_num):
        # 预测这里的逻辑是有大问题的
        if predict([np.array(spatial_test), np.array(temporal_test)], y_test[i]):
            right += 1
    accuracy = (right / sample_num) * 100
    print('预测准确率%f' % accuracy)
    pass