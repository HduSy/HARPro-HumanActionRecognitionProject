import numpy as np

from src.readTxtData2Memory import transformTxtLine2ListObj
from src.utils.utils import generateSpatialFeature, generateTempralAngleFeature

regularization = True  # 归一化与否
frameNum = 25  # n_input time_step步长
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action'
action_type = 'falling'
dataSet = []
spatialN = []
temporalN = []
xn = []
yn = []
space = 1  # 稀疏间隔
m = 0.9  # 训练集测试集划分比例


def readOneActionDataFromTxt(txtDir, action_type):
    global dataSet, spatialN, temporalN, yn, xn
    tmp = []
    spatial_fragment = []
    temporal_fragment = []
    if regularization:
        print('归一化 spatial feature')
    else:
        print('未采用归一化策略')
    print('处理%s动作' % action_type)
    filePath = txtDir + '\\' + action_type + '-result-data.txt'
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
            spatial_fragment.append(sFeature)
            temporal_fragment.append(tFeature)
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
    spatialN = np.array(spatialN)
    temporalN = np.array(temporalN)
    yn = np.array(yn)
    trainsize = int(m * len(dataSet))
    return ((spatialN[:trainsize], temporalN[:trainsize], yn[:trainsize]),
            (spatialN[trainsize:], temporalN[trainsize:], yn[trainsize:]))


if __name__ == '__main__':
    txtDir += '\\' + action_type
