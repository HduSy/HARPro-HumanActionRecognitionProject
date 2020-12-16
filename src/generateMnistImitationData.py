import numpy as np
from time import time
from src.readTxtData2Memory import transformTxtLine2ListObj
from src.utils.utils import generateSpatialFeature, generateTempralAngleFeature, generateTempralLenFeature
from src.utils.utils import fusion, fusionMean, fusionMax, fusionMin
from src.public import actions, txtDir, regularization, anglization
from src.refineAndSaveKeyPointData import write2Txt, write2Txt2

# actions = ['falling1_8', 'falling2_0']
dataSet = []
SFeatures = []
TFeatures = []
space = 1  # 稀疏间隔
frameNum = 25  # n_input time_step步长 TODO:可以做下对比实验啊
# regularization = False  # 归一化与否
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
spatialN_test = []
temporalN_test = []
yn_test = []

# 读取所有动作的数据到dataSet,并打乱以便于存储
def makeDataSetFromTxt(filePath):
    global dataSet
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
                if anglization:
                    tFeature = generateTempralAngleFeature(pre, x, norminalize=False)
                else:
                    tFeature = generateTempralLenFeature(pre, x, norminalize=False)
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
    return dataSet


fileName = None


# 再将持久化数据读到内存
def loadAllShuffledDataFromTxt(filePath):
    with open(filePath) as f:
        line = f.readline()
        # print(line)
        # input()
        while line:
            tmp_arr = line.split(' ')[:-1]
            lineArr = list(map(float, tmp_arr))
            label = line.split(' ')[-1:][0].replace('\n', '')
            # print(lineArr)
            # input()
            sampleSpatial = []
            sampleTemporal = []
            tmp = []
            for i in range(0, 25 * 25):
                tmp.append(lineArr[i])
                if len(tmp) == 25:
                    sampleSpatial.append(np.array(tmp))
                    tmp = []
            tmp = []
            for i in range(25 * 25, 25 * 25 * 2):
                tmp.append(lineArr[i])
                if len(tmp) == 25:
                    sampleTemporal.append(np.array(tmp))
                    tmp = []
            dataSet.append([sampleSpatial, sampleTemporal, label])
            line = f.readline()
            # print(dataSet)
            # input()
        return dataSet


# test:是否是测试集
def spliceDataSet(test=False):
    global trainsize, dataSet, spatialN_test, temporalN_test, yn_test, spatialN, temporalN, yn, fileName
    if regularization:
        if anglization:
            fileName = txtDir + '\\all-shuffled\\all-shuffled-angle-regularized-result-data.txt'
        else:
            fileName = txtDir + '\\all-shuffled\\all-shuffled-len-regularized-result-data.txt'
    else:
        if anglization:
            fileName = txtDir + '\\all-shuffled\\all-shuffled-angle-unregularized-result-data.txt'
        else:
            fileName = txtDir + '\\all-shuffled\\all-shuffled-len-unregularized-result-data.txt'
    dataSet = loadAllShuffledDataFromTxt(fileName)
    # todo:划分训练集、验证集与测试集
    trainsize = int(m * len(dataSet))
    trainsize_t = int(trainsize * m)
    # print(trainsize_t, trainsize, len(dataSet) - trainsize_t)

    # train:[0:int(0.81 * 11337)], test:[int(0.81 * 11337):int(0.9 * 11337)]
    # test:[int(0.9 * 11337):11337]
    # print(len(dataSet))  # 7842*frameNum(25)=196050 196063
    print(len(dataSet))
    print('训练集大小:{0}'.format(trainsize))
    if test:
        # todo:未参与到训练0.1测试集
        dataSet_test = dataSet[trainsize:]
        # np.random.shuffle(dataSet_test)
        print('测试集大小{0}'.format(len(dataSet_test)))
        for k in range(len(dataSet_test)):
            [x1, x2, y] = dataSet_test[k]
            spatialN_test.append(x1)
            temporalN_test.append(x2)
            yn_test.append(y)
        spatialN_test = np.array(spatialN_test)
        temporalN_test = np.array(temporalN_test)
        yn_test = np.array(yn_test)
        return (spatialN_test, temporalN_test, yn_test)
    else:
        # todo:训练集0.9中0.81训练0.09验证
        dataSet_train = dataSet[:trainsize]
        # np.random.shuffle(dataSet_train)
        valisize = int(trainsize * m)
        # train = dataSet[:valisize]
        # vali = dataSet[valisize:trainsize]
        for j in range(len(dataSet_train)):
            [x1, x2, y] = dataSet_train[j]
            spatialN.append(x1)
            temporalN.append(x2)
            yn.append(y)
        spatialN = np.array(spatialN)
        temporalN = np.array(temporalN)
        yn = np.array(yn)
        print('训练集大小:{0},验证集大小:{1}'.format(len(spatialN[:valisize]), len(spatialN[valisize:trainsize])))
        return ((spatialN[:valisize], temporalN[:valisize], yn[:valisize]),
                (spatialN[valisize:trainsize], temporalN[valisize:trainsize], yn[valisize:trainsize]))


# 将打乱后的dataSet持久化
if __name__ == '__main__':
    action_type = 'all-shuffled'
    fileDir = txtDir + '\\' + action_type
    if regularization:
        print('归一化 spatial feature')
        if anglization:
            print('时序角度')
            txtFile = action_type + '-angle-regularized-result-data.txt'
        else:
            print('时序距离')
            txtFile = action_type + '-len-regularized-result-data.txt'
    else:
        print('未采用归一化策略')
        if anglization:
            print('时序角度')
            txtFile = action_type + '-angle-unregularized-result-data.txt'
        else:
            print('时序距离')
            txtFile = action_type + '-len-unregularized-result-data.txt'
    begin = time()
    allShuffledData = makeDataSetFromTxt(txtDir)
    # print(allShuffledData)  # 验证写入与读出的dataSet结构是否一致
    end = time()
    print('程序处理时长约%.1fmin' % ((end - begin) / 60))
    begin = time()  # 开始计时
    # 存入.txt文件
    for i in range(len(allShuffledData)):
        # print(all_frames_effect_keyPointInfo[i])
        [spatialInfo, temporalInfo, actionType] = allShuffledData[i]
        # print(allShuffledData[i])
        # input()
        # print(spatialInfo, temporalInfo, ' is ', actionType)
        # 持久化dataSet
        write2Txt2(fileDir, txtFile, spatialInfo, temporalInfo, actionType)
    end = time()  # 结束计时
    print('写结果用时%.3f' % (end - begin))
    pass
