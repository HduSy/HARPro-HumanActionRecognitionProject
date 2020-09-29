import keras
from src.loadFeatureDataTxt2Mem import loadDataFromTxt

txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action\\all'
spatial_feature_txt = 'all-spatial-feature-5-data.txt'
tempral_feature_txt = 'all-tempral-feature-5-data.txt'
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

n_classes = 6

if __name__ == '__main__':
    spatialFilePath = txtDir + '\\' + spatial_feature_txt
    tempralFilePath = txtDir + '\\' + tempral_feature_txt
    # TODO:当前帧空间相对位置特征&连续5帧序列特征
    (x_train, y_train), (x_test, y_test) = loadDataFromTxt(spatialFilePath, tempralFilePath)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    for i in range(len(y_train)):
        y_train[i] = actions.index(y_train[i])
    for i in range(len(y_test)):
        y_test[i] = actions.index(y_test[i])
    # to_categorical 第一个参数必须是整型数组
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
