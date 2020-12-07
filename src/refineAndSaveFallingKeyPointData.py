import os
from time import *
from src.public import txtDir
actions = ['falling1_8', 'falling2_0']
action_type = 'falling2_0'
saveDir = txtDir + '\\' + action_type
saveTxtFile = action_type + '-result-data.txt'
dataSetDir = 'F:\\XLDownload\\dataSet\\KTH\\' + action_type
person_effect_keyPoints_minCount = 7  # 检测一个骨架是否是一个有效的人
keyPoint_threshold = 0.4  # 关键点置信度
allFramesEffectKeyPointInfo = []

from src.refineAndSaveKeyPointData import readJSON


def readKeyPointJSONFile():
    # 得到当前action所有result-json文件夹路径
    global dataSetDir
    result_json_dirs = os.listdir(dataSetDir)
    jsonFiles = None
    for i in range(len(result_json_dirs)):
        dirName = dataSetDir + '\\' + result_json_dirs[i]
        for root, dirs, files in os.walk(dirName):
            jsonFiles = files
        for j in range(len(jsonFiles)):
            # print(result_json_dirs[i] + '\\' + jsonFiles[j])
            personKeyPointInfo = readJSON(dirName + '\\' + jsonFiles[j])
            if personKeyPointInfo is not None:
                allFramesEffectKeyPointInfo.append([personKeyPointInfo, {'action': action_type}])
                # print(result_json_dirs[i] + '\\' + jsonFiles[j])
            else:
                print('personKeyPointInfo is None 不考虑 ' + result_json_dirs[i] + '\\' + jsonFiles[j])
        # print(len(allFramesEffectKeyPointInfo))  # 360 390 465
    print(allFramesEffectKeyPointInfo)  # (1215, 1, 25)
    # return allFramesEffectKeyPointInfo

from src.refineAndSaveKeyPointData import write2Txt


if __name__ == '__main__':
    # global allFramesEffectKeyPointInfo
    # global saveDir, saveTxtFile
    readKeyPointJSONFile()
    beginTime = time()  # 开始计时
    # 存入.txt文件
    for i in range(len(allFramesEffectKeyPointInfo)):
        # print(allFramesEffectKeyPointInfo[i])
        [keyPointInfo, actionType] = allFramesEffectKeyPointInfo[i]
        # print(keyPointInfo, ' is ', actionType)
        # print(np.array(allFramesEffectKeyPointInfo).shape)  # (1215, 26)
        if keyPointInfo is not None:
            print(keyPointInfo)
            # input()
            write2Txt(saveDir, saveTxtFile, keyPointInfo, actionType)
            # write2Txt(boxing_result_data, keyPointInfo, actionType)
        else:
            print('allFramesEffectKeyPointInfo保存的是所有非None personKeyPointInfo数据,不再会打印这行')
    endTime = time()  # 结束计时
    print('写结果用时%.3f' % (endTime - beginTime))
