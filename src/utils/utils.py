import numpy as np
import os
import math
from string import Template
from sklearn.preprocessing import MinMaxScaler

body_25_key_points_number = 25  # 我们的模型25个关键点
commands_file = '/src/python-lrn/command.txt'
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
angle_pairs = [
    [1, 8, 2, 3],
    [1, 8, 5, 6],
    [1, 8, 3, 4],
    [1, 8, 6, 7],
    [2, 3, 9, 10],
    [2, 3, 10, 11],
    [3, 4, 9, 10],
    [3, 4, 10, 11],
    [5, 6, 12, 13],
    [5, 6, 13, 14],
    [6, 7, 12, 13],
    [6, 7, 13, 14],
    [3, 2, 3, 4],
    [6, 5, 6, 7],
    [8, 1, 9, 10],
    [8, 1, 12, 13],
    [10, 9, 10, 11],
    [13, 12, 13, 14],
    [11, 10, 11, 12],
    [14, 13, 14, 19],
    [2, 3, 5, 6],
    [3, 4, 6, 7],
    [9, 10, 12, 13],
    [10, 11, 13, 14],
    [11, 22, 14, 19]
]


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
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


# 特征融合方法
# weight-fusion
# w1=1, w2=0.1
# LSTM test score: 0.20608791100561238
# LSTM test accuracy: 0.9668790102005005
# LSTM test score: 0.19811472976268832
# LSTM test accuracy: 0.9617834687232971
def fusion(f1, f2, w1=1, w2=0.1):
    return f1 * w1 + f2 * w2


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


# x1,y1,x2,y2,x3,y3,x4,y4
# v1 = [x1,y1,x2,y2] v2 = [x3,y3,x4,y4]
def angle(x1, y1, x2, y2, x3, y3, x4, y4):
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    angle1 = math.atan2(dy1, dx1)
    angle1 = float(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = float(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    # print(included_angle)
    return round(included_angle, 3)


# get personKeyPointList 25 angles
def getPersonKeyPointAngleList(personKeyPointList):
    pair1 = angle(personKeyPointList[1]['x'], personKeyPointList[1]['y'], personKeyPointList[8]['x'],
                  personKeyPointList[8]['y'], personKeyPointList[2]['x'], personKeyPointList[2]['y'],
                  personKeyPointList[3]['x'], personKeyPointList[3]['y'])
    pair2 = angle(personKeyPointList[1]['x'], personKeyPointList[1]['y'], personKeyPointList[8]['x'],
                  personKeyPointList[8]['y'], personKeyPointList[5]['x'], personKeyPointList[5]['y'],
                  personKeyPointList[6]['x'], personKeyPointList[6]['y'])
    pair3 = angle(personKeyPointList[1]['x'], personKeyPointList[1]['y'], personKeyPointList[8]['x'],
                  personKeyPointList[8]['y'], personKeyPointList[3]['x'], personKeyPointList[3]['y'],
                  personKeyPointList[4]['x'], personKeyPointList[4]['y'])
    pair4 = angle(personKeyPointList[1]['x'], personKeyPointList[1]['y'], personKeyPointList[8]['x'],
                  personKeyPointList[8]['y'], personKeyPointList[6]['x'], personKeyPointList[6]['y'],
                  personKeyPointList[7]['x'], personKeyPointList[7]['y'])
    pair5 = angle(personKeyPointList[2]['x'], personKeyPointList[2]['y'], personKeyPointList[3]['x'],
                  personKeyPointList[3]['y'], personKeyPointList[9]['x'], personKeyPointList[9]['y'],
                  personKeyPointList[10]['x'], personKeyPointList[10]['y'])
    pair6 = angle(personKeyPointList[2]['x'], personKeyPointList[2]['y'], personKeyPointList[3]['x'],
                  personKeyPointList[3]['y'], personKeyPointList[10]['x'], personKeyPointList[10]['y'],
                  personKeyPointList[11]['x'], personKeyPointList[11]['y'])
    pair7 = angle(personKeyPointList[3]['x'], personKeyPointList[3]['y'], personKeyPointList[4]['x'],
                  personKeyPointList[4]['y'], personKeyPointList[9]['x'], personKeyPointList[9]['y'],
                  personKeyPointList[10]['x'], personKeyPointList[10]['y'])
    pair8 = angle(personKeyPointList[3]['x'], personKeyPointList[3]['y'], personKeyPointList[4]['x'],
                  personKeyPointList[4]['y'], personKeyPointList[10]['x'], personKeyPointList[10]['y'],
                  personKeyPointList[11]['x'], personKeyPointList[11]['y'])
    pair9 = angle(personKeyPointList[5]['x'], personKeyPointList[5]['y'], personKeyPointList[6]['x'],
                  personKeyPointList[6]['y'], personKeyPointList[12]['x'], personKeyPointList[12]['y'],
                  personKeyPointList[13]['x'], personKeyPointList[13]['y'])
    pair10 = angle(personKeyPointList[5]['x'], personKeyPointList[5]['y'], personKeyPointList[6]['x'],
                   personKeyPointList[6]['y'], personKeyPointList[13]['x'], personKeyPointList[13]['y'],
                   personKeyPointList[14]['x'], personKeyPointList[14]['y'])
    pair11 = angle(personKeyPointList[6]['x'], personKeyPointList[6]['y'], personKeyPointList[7]['x'],
                   personKeyPointList[7]['y'], personKeyPointList[12]['x'], personKeyPointList[12]['y'],
                   personKeyPointList[13]['x'], personKeyPointList[13]['y'])
    pair12 = angle(personKeyPointList[6]['x'], personKeyPointList[6]['y'], personKeyPointList[7]['x'],
                   personKeyPointList[7]['y'], personKeyPointList[13]['x'], personKeyPointList[13]['y'],
                   personKeyPointList[14]['x'], personKeyPointList[14]['y'])
    pair13 = angle(personKeyPointList[3]['x'], personKeyPointList[3]['y'], personKeyPointList[2]['x'],
                   personKeyPointList[2]['y'], personKeyPointList[3]['x'], personKeyPointList[3]['y'],
                   personKeyPointList[4]['x'], personKeyPointList[4]['y'])
    pair14 = angle(personKeyPointList[6]['x'], personKeyPointList[6]['y'], personKeyPointList[5]['x'],
                   personKeyPointList[5]['y'], personKeyPointList[6]['x'], personKeyPointList[6]['y'],
                   personKeyPointList[7]['x'], personKeyPointList[7]['y'])
    pair15 = angle(personKeyPointList[8]['x'], personKeyPointList[8]['y'], personKeyPointList[1]['x'],
                   personKeyPointList[1]['y'], personKeyPointList[9]['x'], personKeyPointList[9]['y'],
                   personKeyPointList[10]['x'], personKeyPointList[10]['y'])
    pair16 = angle(personKeyPointList[8]['x'], personKeyPointList[8]['y'], personKeyPointList[1]['x'],
                   personKeyPointList[1]['y'], personKeyPointList[12]['x'], personKeyPointList[12]['y'],
                   personKeyPointList[13]['x'], personKeyPointList[13]['y'])
    pair17 = angle(personKeyPointList[10]['x'], personKeyPointList[10]['y'], personKeyPointList[9]['x'],
                   personKeyPointList[9]['y'], personKeyPointList[10]['x'], personKeyPointList[10]['y'],
                   personKeyPointList[11]['x'], personKeyPointList[11]['y'])
    pair18 = angle(personKeyPointList[13]['x'], personKeyPointList[13]['y'], personKeyPointList[12]['x'],
                   personKeyPointList[12]['y'], personKeyPointList[13]['x'], personKeyPointList[13]['y'],
                   personKeyPointList[14]['x'], personKeyPointList[14]['y'])
    pair19 = angle(personKeyPointList[11]['x'], personKeyPointList[11]['y'], personKeyPointList[10]['x'],
                   personKeyPointList[10]['y'], personKeyPointList[11]['x'], personKeyPointList[11]['y'],
                   personKeyPointList[12]['x'], personKeyPointList[12]['y'])
    pair20 = angle(personKeyPointList[14]['x'], personKeyPointList[14]['y'], personKeyPointList[13]['x'],
                   personKeyPointList[13]['y'], personKeyPointList[14]['x'], personKeyPointList[14]['y'],
                   personKeyPointList[19]['x'], personKeyPointList[19]['y'])
    pair21 = angle(personKeyPointList[2]['x'], personKeyPointList[2]['y'], personKeyPointList[3]['x'],
                   personKeyPointList[3]['y'], personKeyPointList[5]['x'], personKeyPointList[5]['y'],
                   personKeyPointList[6]['x'], personKeyPointList[6]['y'])
    pair22 = angle(personKeyPointList[3]['x'], personKeyPointList[3]['y'], personKeyPointList[4]['x'],
                   personKeyPointList[4]['y'], personKeyPointList[6]['x'], personKeyPointList[6]['y'],
                   personKeyPointList[7]['x'], personKeyPointList[7]['y'])
    pair23 = angle(personKeyPointList[9]['x'], personKeyPointList[9]['y'], personKeyPointList[10]['x'],
                   personKeyPointList[10]['y'], personKeyPointList[12]['x'], personKeyPointList[12]['y'],
                   personKeyPointList[13]['x'], personKeyPointList[13]['y'])
    pair24 = angle(personKeyPointList[10]['x'], personKeyPointList[10]['y'], personKeyPointList[11]['x'],
                   personKeyPointList[11]['y'], personKeyPointList[13]['x'], personKeyPointList[13]['y'],
                   personKeyPointList[14]['x'], personKeyPointList[14]['y'])
    pair25 = angle(personKeyPointList[11]['x'], personKeyPointList[11]['y'], personKeyPointList[22]['x'],
                   personKeyPointList[22]['y'], personKeyPointList[14]['x'], personKeyPointList[14]['y'],
                   personKeyPointList[19]['x'], personKeyPointList[19]['y'])
    return [pair1, pair2, pair3, pair4, pair5, pair6, pair7, pair8, pair9, pair10, pair11, pair12, pair13, pair14,
            pair15, pair16, pair17, pair18, pair19, pair20,
            pair21, pair22, pair23, pair24, pair25]
    pass


# generate python angle args code
def generateAngleArgsPyCode(p1=None, p2=None, p3=None, p4=None):
    global angle_pairs
    for i in range(1, len(angle_pairs) + 1):
        [p1, p2, p3, p4] = angle_pairs[i - 1]
        tmp = Template(
            "pair${index} = angle(personKeyPointList[${p1}]['x'], personKeyPointList[${p1}]['y'], personKeyPointList[${p2}]['x'],personKeyPointList[${p2}]['y'], personKeyPointList[${p3}]['x'], personKeyPointList[${p3}]['y'],personKeyPointList[${p4}]['x'], personKeyPointList[${p4}]['y'])")
        print(tmp.substitute(index=i, p1=p1, p2=p2, p3=p3, p4=p4))
    # return tmp.substitute(p1=p1, p2=p2, p3=p3, p4=p4)


# 空间分布特征 已验证 norminalize=True 情况下函数正确
def generateSpatialFeature(personKeyPointList, norminalize=False):
    # print('personKeyPointList样子', personKeyPointList)
    # input()
    feature = []
    # 先进行归一化再进行欧氏距离计算
    if norminalize:
        personKeyPointList = keyPointList2List(personKeyPointList)
    MidHip = np.array([personKeyPointList[8]['x'], personKeyPointList[8]['y']])
    for i in range(0, body_25_key_points_number):
        keyPointArray = np.array([personKeyPointList[i]['x'], personKeyPointList[i]['y']])
        EDistance = np.sqrt(np.sum(np.square(keyPointArray - MidHip)))
        feature.append(round(float(EDistance), 3))
    # print(feature)
    return np.array(feature)  # (25,)
    # return feature


# 时间序列特征---距离-受d2缩放影响 已验证 norminalize=True 情况下函数正确
def generateTempralLenFeature(preKeyPointList, curKeyPointList, norminalize=False):
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


# TODO:时间序列特征---角度 可能并不需要归一化，归一化没意义不像spatial feature 25组[x, y]归一化后有意义
def generateTempralAngleFeature(preKeyPointList, curKeyPointList, norminalize=False):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))  # or feature_range=(0, 1)
    feature = None
    preKeyPointAngleList = getPersonKeyPointAngleList(preKeyPointList)
    curKeyPointAngleList = getPersonKeyPointAngleList(curKeyPointList)
    feature = np.array(curKeyPointAngleList) - np.array(preKeyPointAngleList)
    # print(feature.shape)
    if norminalize:
        feature = min_max_scaler.fit_transform(feature.reshape(-1, 1))
        feature = feature.reshape(25, )  # (25,1)
    return feature
