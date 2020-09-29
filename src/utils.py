import numpy as np
import os

body_25_key_points_number = 25  # 我们的模型25个关键点


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        sigm = sigm * (1 - sigm)
    return sigm


def isFloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def warning(msg):
    orange = '\033[33m'
    end = '\033[0m'
    print(orange + msg + end)
    return


def error(msg):
    red = '\033[31m'
    end = '\033[0m'
    print(red + msg + end)
    return


commands_file = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\src\\python-lrn\\command.txt'
command_list = []


def writeFile(fileName):
    global command_list
    if not os.path.exists(fileName):
        f = open(fileName, 'w')
    with open(fileName, 'w', encoding='utf-8') as f:
        for i in range(len(command_list)):
            f.write(command_list[i] + '\n')
        f.close()


def generateUbuntuDirs():
    global command_list
    for i in range(1, 26):
        for j in range(1, 5):
            dirName = 'sudo mkdir p%dd%d' % (i, j)
            command_list.append(dirName)


# win bash批处理copy脚本
# pxdx===>personx_boxing_dx_uncomp
def generateCopyFiles():
    global command_list
    command_list = []
    for i in range(5, 26):
        for j in range(1, 5):
            sourcePath = 'G:\\json\\json\\p%dd%d' % (i, j)
            targetPath = 'F:\\XLDownload\\dataSet\\KTH\\boxing\\boxing-video\\person' + (
                    '%02d' % i) + '_boxing_d%d_uncomp\\' % (j)
            copyCommand = 'xcopy ' + sourcePath + '\\*.*' + ' ' + targetPath + ' ' + '/s /e /c /y /h /r'
            command_list.append(copyCommand)


# 判断文件是否为空，空时写入
def fileIsEmpty(fileName):
    size = os.path.getsize(fileName)
    if size == 0:
        return True
    else:
        return False


# 空间分布特征
def generateSpatialFeature(personKeyPointList):
    feature = []
    MidHip = np.array([personKeyPointList[8]['x'], personKeyPointList[8]['y']])
    for i in range(0, body_25_key_points_number):
        keyPointArray = np.array([personKeyPointList[i]['x'], personKeyPointList[i]['y']])
        EDistance = np.sqrt(np.sum(np.square(keyPointArray - MidHip)))
        feature.append(round(float(EDistance), 3))
    # print(feature)
    return feature


# 时间序列特征
def generateTempralFeature(preKeyPointList, curKeyPointList):
    feature = []
    for i in range(0, body_25_key_points_number):
        preKeyPointArray = np.array([preKeyPointList[i]['x'], preKeyPointList[i]['y']])
        curKeyPointArray = np.array([curKeyPointList[i]['x'], curKeyPointList[i]['y']])
        EDistance = np.sqrt(np.sum(np.square(preKeyPointArray - curKeyPointArray)))
        feature.append(round(float(EDistance), 3))
    # print(feature)
    return feature
