import numpy as np
from time import time
from src.readTxtData2Memory import transformTxtLine2ListObj
from src.utils.utils import generateSpatialFeature, generateTempralAngleFeature, generateTempralLenFeature
from src.utils.utils import fusion, fusionMean, fusionMax, fusionMin
from src.public import actions, txtDir

# actions = ['falling1_8', 'falling2_0']
dataSet = []
SFeatures = []
TFeatures = []
space = 1  # 稀疏间隔
frameNum = 25  # n_input time_step步长 TODO:可以做下对比实验啊
regularization = True  # 归一化与否
# TODO:使用归一化前后对模型准确率的影响
# without regularization
# LSTM test score: 0.19241959800006478
# LSTM test accuracy: 0.9159235954284668
# with spatial feature regularization 提高
# LSTM test score: 0.11868369640058772
# LSTM test accuracy: 0.9541401267051697
# with tempral len feature regularization 稍有降低但基本不变 d2缩放造成的尺度变化对距离特征的影响
# LSTM test score: 0.2668888951088213
# LSTM test accuracy: 0.8955414295196533
# without tempral angle feature regularization
# LSTM test score: 0.15871618061688295
# LSTM test accuracy: 0.9579617977142334
# with tempral angle feature regularization 降低了 因为对角度进行归一化不科学
# LSTM test score: 0.20507974412031235
# LSTM test accuracy: 0.9375796318054199
m = 0.9  # 训练集测试集划分比例
trainsize = None  # 训练集大小
xn = []
yn = []
spatialN = []
temporalN = []


def readDataFromTxt(filePath, test=False):
    global dataSet, xn, yn, trainsize, spatialN, temporalN
    tmp = []
    fragment = []
    spatial_fragment = []
    temporal_fragment = []
    if regularization:
        print('归一化 spatial feature')
    else:
        print('未采用归一化策略')
    # 遍历所有动作数据集
    for index in range(len(actions)):
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
                sFeature = generateSpatialFeature(x, norminalize=regularization)
                tFeature = generateTempralAngleFeature(pre, x, norminalize=False)
                # tFeature = generateTempralLenFeature(pre, x, norminalize=False)
                # print(x)
                # print(sFeature)
                # print(tFeature)
                # input()
                # fragment.append(fusion(sFeature, tFeature, 1, 0.8))
                # fragment.append(fusionMean(sFeature, tFeature))
                # fragment.append(fusionMax(sFeature, tFeature))
                # fragment.append(fusionMin(sFeature, tFeature))
                spatial_fragment.append(sFeature)
                temporal_fragment.append(tFeature)
                # 取frameNum帧作为一个序列样本
                # if len(fragment) == frameNum:
                #     dataSet.append([fragment, y['action']])
                #     fragment = []
                if len(spatial_fragment) == frameNum:
                    dataSet.append([spatial_fragment, temporal_fragment, y['action']])
                    spatial_fragment = []
                    temporal_fragment = []
                pre = x
            i += space
        # 更新tmp&filePath
        filePath = txtDir
        tmp = []
    # shuffle 洗牌打乱所有样本
    np.random.shuffle(dataSet)
    # print(len(dataSet))  # 7842*frameNum(25)=196050 196063
    for i in range(len(dataSet)):
        [x1, x2, y] = dataSet[i]
        spatialN.append(x1)
        temporalN.append(x2)
        yn.append(y)
        # [x, y] = dataSet[i]
        # xn.append(x)
        # yn.append(y)
    # xn = np.array(xn)
    spatialN = np.array(spatialN)
    temporalN = np.array(temporalN)
    yn = np.array(yn)
    trainsize = int(m * len(dataSet))
    trainsize_t = int(trainsize * m)
    # print(trainsize_t, trainsize, len(dataSet) - trainsize_t)

    # train:[0:int(0.81 * 11337)], test:[int(0.81 * 11337):int(0.9 * 11337)]
    # predict:[int(0.9 * 11337):11337]
    if test:
        # 未参与到训练0.1
        return (spatialN[trainsize:], temporalN[trainsize:], yn[trainsize:])
    else:
        # 0.9中0.81训练集0.09测试集
        return ((spatialN[:trainsize_t], temporalN[:trainsize_t], yn[:trainsize_t]),
                (spatialN[trainsize_t:trainsize], temporalN[trainsize_t:trainsize], yn[trainsize_t:trainsize]))
    # return ((xn[:trainsize], yn[:trainsize]), (xn[trainsize:], yn[trainsize:]))
    # return dataSet


if __name__ == '__main__':
    begin = time()
    ((x_train, y_train), (x_test, y_test)) = readDataFromTxt(txtDir)
    end = time()
    print('程序处理时长约%.1fmin' % ((end - begin) / 60))
    # print(x_train, y_train)
    # print(type(x_train), type(y_train))
    # net
    pass
