import numpy as np
import os
import cmath
from sklearn.preprocessing import MinMaxScaler

body_25_key_points_number = 25  # 我们的模型25个关键点
commands_file = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\src\\python-lrn\\command.txt'
command_list = []
frame01 = [{'x': 76, 'y': 22, 'score': 0.852835}, {'x': 82, 'y': 32, 'score': 0.754131},
           {'x': 80, 'y': 31, 'score': 0.645748}, {'x': 70, 'y': 36, 'score': 0.717755},
           {'x': 61, 'y': 35, 'score': 0.818968}, {'x': 84, 'y': 33, 'score': 0.789034},
           {'x': 81, 'y': 45, 'score': 0.761079}, {'x': 74, 'y': 41, 'score': 0.778467},
           {'x': 79, 'y': 58, 'score': 0.601553}, {'x': 79, 'y': 58, 'score': 0.425949},
           {'x': 79, 'y': 79, 'score': 0.603427}, {'x': 79, 'y': 100, 'score': 0.570084},
           {'x': 79, 'y': 59, 'score': 0.615698}, {'x': 74, 'y': 79, 'score': 0.677094},
           {'x': 79, 'y': 102, 'score': 0.638697}, {'x': 76, 'y': 21, 'score': 0.121608},
           {'x': 77, 'y': 21, 'score': 0.796632}, {'x': 0, 'y': 0, 'score': 0.0}, {'x': 82, 'y': 22, 'score': 0.892339},
           {'x': 70, 'y': 103, 'score': 0.641912}, {'x': 71, 'y': 105, 'score': 0.632404},
           {'x': 80, 'y': 105, 'score': 0.619728}, {'x': 70, 'y': 101, 'score': 0.410267},
           {'x': 71, 'y': 100, 'score': 0.369467}, {'x': 81, 'y': 102, 'score': 0.178743}]
frame02 = [{'score': 0.309807, 'x': 81, 'y': 22}, {'score': 0.6664, 'x': 90, 'y': 30},
           {'score': 0.523754, 'x': 92, 'y': 28}, {'score': 0.211789, 'x': 81, 'y': 40},
           {'score': 0.301164, 'x': 71, 'y': 35}, {'score': 0.71309, 'x': 90, 'y': 30},
           {'score': 0.714676, 'x': 79, 'y': 39}, {'score': 0.786523, 'x': 69, 'y': 32},
           {'score': 0.605353, 'x': 87, 'y': 55}, {'score': 0.492125, 'x': 85, 'y': 55},
           {'score': 0.621014, 'x': 83, 'y': 75}, {'score': 0.750422, 'x': 84, 'y': 92},
           {'score': 0.620267, 'x': 89, 'y': 56}, {'score': 0.653355, 'x': 87, 'y': 76},
           {'score': 0.686467, 'x': 89, 'y': 95}, {'score': 0.0, 'x': 0, 'y': 0},
           {'score': 0.384614, 'x': 82, 'y': 20}, {'score': 0.0, 'x': 0, 'y': 0},
           {'score': 0.578789, 'x': 86, 'y': 21}, {'score': 0.645289, 'x': 80, 'y': 97},
           {'score': 0.582743, 'x': 82, 'y': 99}, {'score': 0.635565, 'x': 90, 'y': 98},
           {'score': 0.640339, 'x': 77, 'y': 95}, {'score': 0.599087, 'x': 77, 'y': 94},
           {'score': 0.382946, 'x': 84, 'y': 94}]


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


# 计算平面坐标系内两点间欧式距离
def euclidDistance(x1, y1, x2, y2):
    return cmath.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


# 特征融合方法
def fusion(f1, f2, w1, w2):
    return np.array(f1) * w1 + np.array(f2) * w2


# 将[{'x':76,'y':22,'score':0.852835},{},{}...]转化为[[76,22],[],[]...] or [76,22,......] (前者吧 x、y两维分别独立归一化)
def keyPointList2List(personKeyPointList):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    temp = []
    result = []
    for i in range(body_25_key_points_number):
        keyPoint = personKeyPointList[i]
        temp.append([keyPoint['x'], keyPoint['y']])
    temp_min_max = min_max_scaler.fit_transform(np.array(temp))
    [rows, columns] = temp_min_max.shape
    for i in range(rows):
        result.append({'x': temp_min_max[i][0], 'y': temp_min_max[i][1]})
    # print(result)
    return result


# 空间分布特征 已验证 norminalize=True 情况下函数正确
def generateSpatialFeature(personKeyPointList, norminalize=False):
    # print('personKeyPointList样子', personKeyPointList)
    # input()
    feature = []
    if norminalize:
        personKeyPointList = keyPointList2List(personKeyPointList)
    MidHip = np.array([personKeyPointList[8]['x'], personKeyPointList[8]['y']])
    for i in range(0, body_25_key_points_number):
        keyPointArray = np.array([personKeyPointList[i]['x'], personKeyPointList[i]['y']])
        EDistance = np.sqrt(np.sum(np.square(keyPointArray - MidHip)))
        feature.append(round(float(EDistance), 3))
    # print(feature)
    return feature


# 时间序列特征 已验证 norminalize=True 情况下函数正确
def generateTempralFeature(preKeyPointList, curKeyPointList, norminalize=False):
    feature = []
    if norminalize:
        preKeyPointList = keyPointList2List(preKeyPointList)
        curKeyPointList = keyPointList2List(curKeyPointList)
    for i in range(0, body_25_key_points_number):
        preKeyPointArray = np.array([preKeyPointList[i]['x'], preKeyPointList[i]['y']])
        curKeyPointArray = np.array([curKeyPointList[i]['x'], curKeyPointList[i]['y']])
        EDistance = np.sqrt(np.sum(np.square(preKeyPointArray - curKeyPointArray)))
        feature.append(round(float(EDistance), 3))
    # print(feature)
    return feature
