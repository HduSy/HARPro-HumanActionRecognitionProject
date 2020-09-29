import json
import os
import numpy as np

# baseDir = '../result-json/'
keyPointDir = '../result-json/'
keyPointFilePrefix = 'person01_walking_d1_uncomp_'
keyPointFileName = None
frameIndex = 0
frameCount = 555
body_25_key_points_number = 25  # 我们的模型25个关键点
person_effect_keyPoints_minCount = 10  # 检测一个骨架是否是一个有效的人
keyPoint_threshold = 0.4  # 关键点置信度
SFeatures = []
TFeatures = []
POSE_BODY_25_BODY_PARTS = [
    {0, "鼻子"},
    {1, "脖子"},
    {2, "右肩"},
    {3, "右肘"},
    {4, "右手腕"},
    {5, "左肩"},
    {6, "左肘"},
    {7, "左手腕"},
    {8, "胯中心"},
    {9, "右臀"},
    {10, "右膝"},
    {11, "右踝"},
    {12, "左跨"},
    {13, "左膝"},
    {14, "左踝"},
    {15, "右眼"},
    {16, "左眼"},
    {17, "右耳"},
    {18, "左耳"},
    {19, "左脚内"},
    {20, "左脚外"},
    {21, "左脚跟"},
    {22, "右脚内"},
    {23, "右脚外"},
    {24, "右脚跟"},
    {25, "Background"},
]


def getKeyPointFileName(idx):
    keyPointFileName = keyPointDir + keyPointFilePrefix + ('%012d' % idx) + '_keypoints.json'
    return keyPointFileName


def fileIsEmpty(fileName):
    size = os.path.getsize(fileName)
    if size == 0:
        return True
    else:
        return False

# 多目标，对openpose识别为多目标情况下每个目标做处理
def getKeyPointInfo(keyPointFileName):
    keyPointsInfo = []
    if os.path.exists(keyPointFileName) and (not fileIsEmpty(keyPointFileName)):
        contexts = open(keyPointFileName, "r", encoding='UTF-8')
        hjson = json.load(contexts)
        # print(hjson)
        people = hjson['people']
        # 处理每一个人的信息
        for onePerson in people:
            pose_keypoints_2d = onePerson['pose_keypoints_2d']
            keyPointsCount = len(pose_keypoints_2d) // 3
            keyPointsList = []
            for i in range(0, keyPointsCount):
                oneKeyPointInfo = {}
                oneKeyPointInfo['x'] = round(float(pose_keypoints_2d[3 * i + 0]))  # round不加第二个参数只保留整数部分
                oneKeyPointInfo['y'] = round(float(pose_keypoints_2d[3 * i + 1]))
                oneKeyPointInfo['score'] = float(pose_keypoints_2d[3 * i + 2])
                keyPointsList.append(oneKeyPointInfo)
            keyPointsInfo.append(keyPointsList)
            # print(keyPointsInfo)
        contexts.close()
    return keyPointsInfo


def getRefinedPersonKeyPointsInfo(keyPointsInfo):
    refinedKeyPointsInfo = []
    peopleCounts = len(keyPointsInfo)
    if peopleCounts == 1:
        effectKeyPointCounts = 0
        keyPointsList = keyPointsInfo[0]
        for i in range(0, 25):
            if keyPointsList[i]['score'] <= keyPoint_threshold:
                continue
            effectKeyPointCounts = effectKeyPointCounts + 1
        # print(effectKeyPointCounts)
        if effectKeyPointCounts >= person_effect_keyPoints_minCount:
            refinedKeyPointsInfo.append(keyPointsInfo[0])
    return refinedKeyPointsInfo

# 单目标，只处理openpose识别为单人情况
def getKeyPointInfo2(keyPointFileName):
    keyPointsInfo = []
    if os.path.exists(keyPointFileName) and (not fileIsEmpty(keyPointFileName)):
        contexts = open(keyPointFileName, "r", encoding='UTF-8')
        hjson = json.load(contexts)
        people = hjson['people']
        # print(len(people))
        if len(people) == 1:
            keyPointsList = []
            pose_keypoints_2d = people[0]['pose_keypoints_2d']
            keyPointsCount = len(pose_keypoints_2d) // 3
            for i in range(0, keyPointsCount):
                oneKeyPointInfo = {}
                oneKeyPointInfo['x'] = round(float(pose_keypoints_2d[3 * i + 0]))  # round不加第二个参数只保留整数部分
                oneKeyPointInfo['y'] = round(float(pose_keypoints_2d[3 * i + 1]))
                oneKeyPointInfo['score'] = float(pose_keypoints_2d[3 * i + 2])
                keyPointsList.append(oneKeyPointInfo)
            keyPointsInfo.append(keyPointsList)
        contexts.close()
    return keyPointsInfo


def getKeyPointInfoArray(keyPointFileName):
    keyPointsInfo = []
    if os.path.exists(keyPointFileName) and (not fileIsEmpty(keyPointFileName)):
        contexts = open(keyPointFileName, "r", encoding='UTF-8')
        hjson = json.load(contexts)
        people = hjson['people']
        # print(len(people))
        if len(people) == 1:
            keyPointsList = []
            pose_keypoints_2d = people[0]['pose_keypoints_2d']
            keyPointsCount = len(pose_keypoints_2d) // 3
            for i in range(0, keyPointsCount):
                x = round(float(pose_keypoints_2d[3 * i + 0]))
                y = round(float(pose_keypoints_2d[3 * i + 1]))
                score = float(pose_keypoints_2d[3 * i + 2])
                oneKeyPointInfo = [x, y, score]
                keyPointsList.append(oneKeyPointInfo)
            keyPointsInfo.append(keyPointsList)
        contexts.close()
    return keyPointsInfo


def readJson():
    frameIndex = 1
    while frameIndex < frameCount:
        preKeyPointFileName = getKeyPointFileName(frameIndex - 1)
        keyPointFileName = getKeyPointFileName(frameIndex)
        # personsKeyPointInfo = getKeyPointInfo2(keyPointFileName)
        preKeyPointInfo = getRefinedPersonKeyPointsInfo(getKeyPointInfo2(preKeyPointFileName))
        personsKeyPointInfo = getRefinedPersonKeyPointsInfo(getKeyPointInfo2(keyPointFileName))
        # print(personsKeyPointInfo)
        if len(personsKeyPointInfo) == 1 and len(preKeyPointInfo) == 1:
            print(personsKeyPointInfo[0])
            SFeatures.append(spatialFeature(personsKeyPointInfo[0]))
            TFeatures.append(tempralFeature(preKeyPointInfo[0], personsKeyPointInfo[0]))
        frameIndex += 1
    return [np.array(SFeatures), np.array(TFeatures)]
    # print(np.array(SFeatures))
    # print(np.array(TFeatures))


def spatialFeature(personKeyPointList):
    feature = []
    MidHip = np.array([personKeyPointList[8]['x'], personKeyPointList[8]['y']])
    for i in range(0, body_25_key_points_number):
        keyPointArray = np.array([personKeyPointList[i]['x'], personKeyPointList[i]['y']])
        EDistance = np.sqrt(np.sum(np.square(keyPointArray - MidHip)))
        feature.append(round(float(EDistance), 3))
    # print(feature)
    return feature


def tempralFeature(preKeyPointList, curKeyPointList):
    feature = []
    for i in range(0, body_25_key_points_number):
        preKeyPointArray = np.array([preKeyPointList[i]['x'], preKeyPointList[i]['y']])
        curKeyPointArray = np.array([curKeyPointList[i]['x'], curKeyPointList[i]['y']])
        EDistance = np.sqrt(np.sum(np.square(preKeyPointArray - curKeyPointArray)))
        feature.append(round(float(EDistance), 3))
    # print(feature)
    return feature


test1 = [{'x': 128, 'y': 27, 'score': 0.52877}, {'x': 120, 'y': 38, 'score': 0.664455},
         {'x': 120, 'y': 38, 'score': 0.766223}, {'x': 120, 'y': 53, 'score': 0.755534},
         {'x': 126, 'y': 63, 'score': 0.757876}, {'x': 118, 'y': 36, 'score': 0.519313},
         {'x': 0, 'y': 0, 'score': 0.0}, {'x': 0, 'y': 0, 'score': 0.0},
         {'x': 120, 'y': 65, 'score': 0.584436},  # 第8个关键点 中心节点
         {'x': 120, 'y': 65, 'score': 0.559692},
         {'x': 116, 'y': 84, 'score': 0.779201}, {'x': 98, 'y': 97, 'score': 0.75832},
         {'x': 123, 'y': 65, 'score': 0.528051}, {'x': 128, 'y': 83, 'score': 0.74762},
         {'x': 128, 'y': 102, 'score': 0.771071}, {'x': 127, 'y': 25, 'score': 0.604091},
         {'x': 0, 'y': 0, 'score': 0.0}, {'x': 123, 'y': 27, 'score': 0.792117}, {'x': 0, 'y': 0, 'score': 0.0},
         {'x': 137, 'y': 105, 'score': 0.708141}, {'x': 136, 'y': 104, 'score': 0.630206},
         {'x': 126, 'y': 105, 'score': 0.686246}, {'x': 99, 'y': 105, 'score': 0.664158},
         {'x': 97, 'y': 103, 'score': 0.603612}, {'x': 95, 'y': 96, 'score': 0.665465}]
test2 = [{'x': 130, 'y': 25, 'score': 0.37792}, {'x': 123, 'y': 37, 'score': 0.641717},
         {'x': 123, 'y': 38, 'score': 0.755243}, {'x': 123, 'y': 53, 'score': 0.738977},
         {'x': 127, 'y': 64, 'score': 0.765292}, {'x': 120, 'y': 36, 'score': 0.494888},
         {'x': 0, 'y': 0, 'score': 0.0}, {'x': 0, 'y': 0, 'score': 0.0},
         {'x': 123, 'y': 64, 'score': 0.581957},  # 第8个关键点 中心节点
         {'x': 123, 'y': 64, 'score': 0.612714},
         {'x': 121, 'y': 84, 'score': 0.760817}, {'x': 102, 'y': 95, 'score': 0.750861},
         {'x': 125, 'y': 64, 'score': 0.457258}, {'x': 130, 'y': 83, 'score': 0.642034},
         {'x': 128, 'y': 102, 'score': 0.72944}, {'x': 129, 'y': 25, 'score': 0.49147}, {'x': 0, 'y': 0, 'score': 0.0},
         {'x': 125, 'y': 27, 'score': 0.784945}, {'x': 0, 'y': 0, 'score': 0.0},
         {'x': 137, 'y': 105, 'score': 0.719679}, {'x': 136, 'y': 104, 'score': 0.63371},
         {'x': 126, 'y': 105, 'score': 0.647789}, {'x': 102, 'y': 105, 'score': 0.640927},
         {'x': 100, 'y': 103, 'score': 0.525561}, {'x': 100, 'y': 95, 'score': 0.660238}]


def main():
    [SFeatures, TFeatures] = readJson()
    # label = np.array(['walking' for i in range(SFeatures.shape[0])])
    # print(SFeatures.shape)  # (233,25)
    # print(TFeatures.shape)  # (233,25)
    # print(label)  # (233,)
    # print(spatialFeature(test1))
    # tempralFeature(test1, test2)


if __name__ == '__main__':
    main()
