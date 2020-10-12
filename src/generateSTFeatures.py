import os
from src.readTxtData2Memory import readDataFromTxt
from src.readTxtData2Memory import txtDir, dataSet
from src.utils.utils import generateSpatialFeature, generateTempralLenFeature

# actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
action_type = 'handclapping'
space = 5
fileDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action\\' + action_type
txtSpatialDataFile = action_type + '-spatial-feature-%d-data.txt' % space
txtTempralDataFile = action_type + '-tempral-feature-%d-data.txt' % space
SFeatures = []
TFeatures = []


def write2Txt(fileDir, fileName, list25, action):
    if not os.path.exists(fileDir):
        os.mkdir(fileDir)
    filePath = fileDir + '\\' + fileName
    # 防止重复追加写入
    fileObj = open(filePath, mode='a', encoding='utf-8')
    line = ''
    for i in range(25):
        line += str(list25[i]) + ' '
    line += action + '\n'
    # print(line)
    fileObj.write(line)


if __name__ == '__main__':
    global actions
    readDataFromTxt(txtDir, False, action_type)
    i = 0
    # TODO:稀疏采样
    while i < len(dataSet):
        [x, y] = dataSet[i]
        if i == 0:
            pre = x
        else:
            # 同步取同一帧的空间分布特征和时间序列特征
            sFeature = generateSpatialFeature(x)
            tFeature = generateTempralLenFeature(pre, x)
            # SFeatures.append(sFeature)
            # TFeatures.append(tFeature)
            write2Txt(fileDir, txtSpatialDataFile, sFeature, y['action'])
            write2Txt(fileDir, txtTempralDataFile, tFeature, y['action'])
            pre = x
        i += space
    # print(len(SFeatures) == len(TFeatures))
