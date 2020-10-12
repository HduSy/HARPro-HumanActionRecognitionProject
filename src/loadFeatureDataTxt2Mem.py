import numpy as np
from src.utils.utils import fusion
txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action\\all'
x_train = []
x_test = []
y_train = []
y_test = []
spatial_feature_txt = 'all-spatial-feature-5-data.txt'
tempral_feature_txt = 'all-tempral-feature-5-data.txt'
spatialFeatureList = []
tempralFeatureList = []
fusionFeature = []
w1 = 1  # 特征融合参数1
w2 = 2  # 特征融合参数2
actionLabel = []


def transformTxtLine2FloatList(line):
    return list(map(float, line.split(' ')[:-1]))


def loadDataFromTxt(sFilePath, tFilePath):
    x, x1, y, y1 = 1, 2, 3, 4
    # from txt to feature_list
    global spatialFeatureList, tempralFeatureList, actionLabel, fusionFeature
    sAction = []
    tAction = []
    all2shuffle = []
    with open(sFilePath) as fileObj:
        line = fileObj.readline()
        while line:
            spatialFeatureList.append(list(map(float, line.split(' ')[:-1])))  # 39208
            sAction.append(line.strip('\n').split(' ')[25])
            line = fileObj.readline()
        fileObj.close()
    with open(tFilePath) as fileObj:
        line = fileObj.readline()
        while line:
            tempralFeatureList.append(list(map(float, line.split(' ')[:-1])))  # 39208
            tAction.append(line.strip('\n').split(' ')[25])
            line = fileObj.readline()
        fileObj.close()
    # 验证时空特征数据对应类别标签对应与否
    for i in range(len(sAction)):
        if sAction[i] == tAction[i]:
            continue
        print('not the correspondence,ops')
    for i in range(len(spatialFeatureList)):
        all2shuffle.append([spatialFeatureList[i], tempralFeatureList[i], sAction[i]])
    # print(len(all2shuffle))  # 39208
    # print(all2shuffle[0])
    # shuffle
    np.random.shuffle(all2shuffle)
    spatialFeatureList = []
    tempralFeatureList = []
    for i in range(len(all2shuffle)):
        spatialFeatureList.append(all2shuffle[i][0])
        tempralFeatureList.append(all2shuffle[i][1])
        actionLabel.append(all2shuffle[i][2])
    # fusion
    for i in range(len(all2shuffle)):
        fusionFeature.append(fusion(spatialFeatureList[i], tempralFeatureList[i], w1, w2))
    fusionFeature = np.array(fusionFeature)
    actionLabel = np.array(actionLabel)
    # slice
    return ((fusionFeature[:30000], actionLabel[:30000]), (fusionFeature[30000:], actionLabel[30000:]))


# 简单时空特征提取
if __name__ == '__main__':
    spatialFilePath = txtDir + '\\' + spatial_feature_txt
    tempralFilePath = txtDir + '\\' + tempral_feature_txt
    (x_train, x_test), (y_train, y_test) = loadDataFromTxt(spatialFilePath, tempralFilePath)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
