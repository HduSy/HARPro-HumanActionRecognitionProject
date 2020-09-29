import numpy as np
from time import time
from src.readTxtData2Memory import transformTxtLine2ListObj
from src.utils import generateSpatialFeature, generateTempralFeature
from src.loadFeatureDataTxt2Mem import fusion

actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action'
dataSet = []
SFeatures = []
TFeatures = []
space = 1  # 稀疏间隔
frameNum = 25
xn = []
yn = []


def readDataFromTxt(filePath):
    global txtDir, actions, dataSet, xn, yn
    tmp = []
    fragment = []
    # 遍历所有动作数据集
    for index in range(6):
        print('处理%s动作' % actions[index])
        filePath += '\\' + actions[index] + '\\' + actions[index] + '-result-data.txt'
        # 取一种动作
        with open(filePath, 'r') as fileObj:
            line = fileObj.readline()
            while line:
                tmp.append(transformTxtLine2ListObj(line.strip('\n')))
                line = fileObj.readline()
            fileObj.close()
        # 处理dataSet
        print(len(tmp))
        i = 0
        while i < len(tmp):
            [x, y] = tmp[i]
            if i == 0:
                pre = x
            else:
                # 同步取同一帧的空间分布特征和时间序列特征
                sFeature = generateSpatialFeature(x)
                tFeature = generateTempralFeature(pre, x)
                # TODO:fusion存的是np.ndarray
                fragment.append(fusion(sFeature, tFeature, 1, 2))
                # 取frameNum帧作为一个序列样本
                if len(fragment) == frameNum:
                    dataSet.append([fragment, y['action']])
                    fragment = []
                pre = x
            i += space
        # 更新tmp&filePath
        filePath = txtDir
        tmp = []
    # shuffle 洗牌打乱所有样本
    np.random.shuffle(dataSet)
    # print(len(dataSet))  # 7842*frameNum(25)=196050 196063
    for i in range(len(dataSet)):
        [x, y] = dataSet[i]
        xn.append(x)
        yn.append(y)
    xn = np.array(xn)
    yn = np.array(yn)
    return ((xn[:7000], yn[:7000]), (xn[7000:], yn[7000:]))
    # return dataSet


if __name__ == '__main__':
    begin = time()
    ((x_train, y_train), (x_test, y_test)) = readDataFromTxt(txtDir)
    end = time()
    print('程序处理时长%fs' % (end - begin))
    # print(x_train, y_train)
    # print(type(x_train), type(y_train))
    # net
    pass
