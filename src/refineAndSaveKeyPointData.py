import os
import json
from time import *
from src.public import txtDir, frameNum, inputA_dim, inputB_dim, anglization

# actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking', 'falling1_8', 'falling2_0']
action_type = 'falling2_0'
# TODO:对每种动作分别按训练集、测试集、验证集划分，这样predict时可按需加载
train0_81 = 0.81
test0__9 = 0.09
val0_1 = 0.1
dataSetDir = 'F:\\XLDownload\\dataSet\\KTH\\' + action_type + '\\' + action_type + '-video'
boxing_result_data = txtDir + '\\' + action_type + '\\' + action_type + '-result-data.txt'
fileDir = txtDir + '\\' + action_type
txtFile = action_type + '-result-data.txt'

person_effect_keyPoints_minCount = 10  # 检测一个骨架是否是一个有效的人
keyPoint_threshold = 0.4  # 关键点置信度
all_frames_effect_keyPointInfo = []


#  获取特定动作json数据所在文件夹 result-json:
#  F:\\XLDownload\\dataSet\\KTH\\action_type\\action_type-video\\personx_boxing_dx_uncomp\\result-json
def traverseJSONDirs(dataDir):
    personx_action_dx_uncomps = os.listdir(dataDir)
    dirs = []
    result_json_dirs = []
    for i in range(len(personx_action_dx_uncomps)):
        dir = personx_action_dx_uncomps[i]
        dirs.append(dir)
        # F:\XLDownload\dataSet\KTH\handclapping\handclapping-video\person01_handclapping_d1_uncomp\result-json
        if os.listdir(dataSetDir + '\\' + dir) == 1:
            result_json_dir = os.listdir(dataSetDir + '\\' + dir)[0]
        # print(result_json_dir)
    for i in range(len(dirs)):
        # F:\\XLDownload\\dataSet\\KTH\\boxing\\boxing-video\\person01_boxing_d1_uncomp\\result-json
        result_json_path = dataSetDir + '\\' + dirs[i] + '\\result-json'
        # print(result_json_path)
        result_json_dirs.append(result_json_path)
    return result_json_dirs


def generateKeyPointFileName(result_json_dir, idx):
    json_file_prefix = result_json_dir.split('\\')[6]
    keyPointFileName = json_file_prefix + ('%012d' % idx) + '_keypoints.json'
    return keyPointFileName


# 生成文件夹 personx_actionType_dx_uncomp
def generateDirs():
    dirsNames = []
    for i in range(1, 26):
        for j in range(1, 5):
            dirName = 'person' + ('%02d' % i) + '_' + action_type + '_d' + ('%d' % j) + '_uncomp'
            # print(dirName)
            dirsNames.append(dirName)
    for i in range(len(dirsNames)):
        if not os.path.exists(dataSetDir + '\\' + dirsNames[i]):
            os.mkdir(dataSetDir + '\\' + dirsNames[i])


def fileIsEmpty(fileName):
    size = os.path.getsize(fileName)
    if size == 0:
        return True
    else:
        return False


# F:\\XLDownload\\dataSet\\KTH\\boxing\\boxing-video\\person02_boxing_d2_uncomp\\result-json\\person02_boxing_d2_uncomp_000000000263_keypoints.json
# 指定json文件路径，从json文件中读取关键点数据
def getKeyPointInfo2(keyPointFileName):
    keyPointsInfo = []
    if os.path.exists(keyPointFileName) and (not fileIsEmpty(keyPointFileName)):
        contexts = open(keyPointFileName, "r", encoding='UTF-8')
        hjson = json.load(contexts)
        people = hjson['people']
        # TODO:p2d2检测出两个人 len(people)=2,直接取第一个了没有做处理
        # print(len(people))
        # if len(people) >= 1:
        #     keyPointsList = []
        #     pose_keypoints_2d = people[0]['pose_keypoints_2d']
        #     keyPointsCount = len(pose_keypoints_2d) // 3
        #     for i in range(0, keyPointsCount):
        #         oneKeyPointInfo = {}
        #         oneKeyPointInfo['x'] = round(float(pose_keypoints_2d[3 * i + 0]))  # round不加第二个参数只保留整数部分
        #         oneKeyPointInfo['y'] = round(float(pose_keypoints_2d[3 * i + 1]))
        #         oneKeyPointInfo['score'] = float(pose_keypoints_2d[3 * i + 2])
        #         keyPointsList.append(oneKeyPointInfo)
        #     keyPointsInfo.append(keyPointsList)
        for onePeople in people:
            pose_keypoints_2d = onePeople['pose_keypoints_2d']
            keyPointsCount = len(pose_keypoints_2d) // 3
            keyPointsList = []
            for i in range(0, keyPointsCount):
                oneKeyPointInfo = {}
                oneKeyPointInfo['x'] = round(float(pose_keypoints_2d[3 * i + 0]))  # round不加第二个参数只保留整数部分
                oneKeyPointInfo['y'] = round(float(pose_keypoints_2d[3 * i + 1]))
                oneKeyPointInfo['score'] = float(pose_keypoints_2d[3 * i + 2])
                keyPointsList.append(oneKeyPointInfo)
            keyPointsInfo.append(keyPointsList)
        contexts.close()
    return keyPointsInfo


# 计算有效关键点数
def effectKeyPointNum(keyPoints25):
    effectCounts = 0
    for i in range(25):
        if keyPoints25[i]['score'] <= keyPoint_threshold:
            continue
        effectCounts += 1
    return effectCounts


# TODO:multiple_people 当由于openpose噪音识别错误出现两个目标或多个目标时选择有效关键点较多那个
def makeChoiceWhichPeople(peopleList):
    target = peopleList[0]
    max = effectKeyPointNum(peopleList[0])
    for i in range(1, len(peopleList)):
        tmp = effectKeyPointNum(peopleList[i])
        # 若是等于怎么处理
        if tmp > max:
            max = tmp
            target = peopleList[i]
    return target


'''
input:从关键点数据json文件中提取的people格式化关键点数据
condition:未检测到人时,return None;检测结果恰有一人时,正常处理;检测结果中多人时,取有效关键点最多那个目标为target.
'''


# TODO:有效关键点个数大于阈值判断为有效识别目标
def getRefinedPersonKeyPointsInfo(keyPointsInfo):
    refinedKeyPointsInfo = []
    target = []
    if len(keyPointsInfo) < 1:
        # return refinedKeyPointsInfo
        return target
    if len(keyPointsInfo) == 1:
        # input('按单目标处理')
        target = keyPointsInfo[0]
    else:
        # input('选择有效关键点最多那个')
        target = makeChoiceWhichPeople(keyPointsInfo)

    effectKeyPointCounts = 0
    for i in range(0, 25):
        if target[i]['score'] <= keyPoint_threshold:
            continue
        effectKeyPointCounts = effectKeyPointCounts + 1
    # print(effectKeyPointCounts)
    # print(target)
    if effectKeyPointCounts >= person_effect_keyPoints_minCount:
        refinedKeyPointsInfo.append(target)
    return refinedKeyPointsInfo


# refined后的关键点信息有可能因为不满足有效关键点个数>10返回结果为None
def readJSON(filePath):
    personsKeyPointInfo = getRefinedPersonKeyPointsInfo(getKeyPointInfo2(filePath))
    # print(personsKeyPointInfo)
    # 存在识别出一个以上情况,getRefinedPersonKeyPointsInfo函数中已做过选择
    if len(personsKeyPointInfo) > 0:
        personsKeyPointInfo[0].append({'action': action_type})  # 增加了对应动作类别,增不增加无所谓可能多余
        return personsKeyPointInfo[0]  # 单人
    else:
        # 有效关键点个数不满足大于10，不将其识别为有效目标
        return None


# 遍历action_type-video文件夹下所有personx_actionType_dx_uncomp文件夹中所有.json文件
def readKeyPointJSONFile():
    # global all_frames_effect_keyPointInfo
    # 得到当前action所有result-json文件夹路径
    result_json_dirs = traverseJSONDirs(dataSetDir)
    jsonFiles = None
    for i in range(len(result_json_dirs)):
        for root, dirs, files in os.walk(result_json_dirs[i]):
            jsonFiles = files
        for j in range(len(jsonFiles)):
            # print(result_json_dirs[i] + '\\' + jsonFiles[j])
            personKeyPointInfo = readJSON(result_json_dirs[i] + '\\' + jsonFiles[j])
            if personKeyPointInfo is not None:
                all_frames_effect_keyPointInfo.append([personKeyPointInfo, {'action': action_type}])
                # print(result_json_dirs[i] + '\\' + jsonFiles[j])
            else:
                print('personKeyPointInfo is None 不考虑 ' + result_json_dirs[i] + '\\' + jsonFiles[j])
        # print(len(all_frames_effect_keyPointInfo))  # 360 390 465
    print(all_frames_effect_keyPointInfo)  # (1215, 1, 25)
    # return all_frames_effect_keyPointInfo


# 将关键点结果及动作标签写入txt
def write2Txt(fileDir, fileName, x, y):
    if not os.path.exists(fileDir):
        os.mkdir(fileDir)
    fileObj = open(fileDir + '\\' + fileName, mode='a', encoding="utf-8")
    line = ''
    for i in range(25):
        line += str(x[i]['x']) + ' ' + str(x[i]['y']) + ' ' + str(x[i]['score']) + ' '
    # print(line)
    line += y['action'] + '\n'
    fileObj.write(line)
    # fileObj.close()


# 将shuffled dataSet持久化
def write2Txt2(fileDir, fileName, x1, x2, y):
    if not os.path.exists(fileDir):
        os.mkdir(fileDir)
    fileObj = open(fileDir + '\\' + fileName, mode='a', encoding="utf-8")
    line = ''
    for i in range(frameNum):
        for j in range(inputA_dim):
            line += str(x1[i][j]) + ' '
    for i in range(frameNum):
        for j in range(inputB_dim):
            line += str(x2[i][j]) + ' '
    line += y + '\n'
    # print(line)
    fileObj.write(line)


# 将关键点结果及动作标签写入txt
# def write2Txt(filePath, x, y):
#     fileDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action\\' + action_type
#     txtFile = action_type + '-result-data.txt'
#     if not os.path.exists(fileDir):
#         os.mkdir(fileDir)
#     fileObj = open(fileDir + '\\' + txtFile, mode='a', encoding="utf-8")
#     line = ''
#     for i in range(25):
#         line += str(x[i]['x']) + ' ' + str(x[i]['y']) + ' ' + str(x[i]['score']) + ' '
#     # print(line)
#     line += y['action'] + '\n'
#     fileObj.write(line)


if __name__ == '__main__':
    # global all_frames_effect_keyPointInfo
    # global fileDir, txtFile
    readKeyPointJSONFile()
    beginTime = time()  # 开始计时
    # 存入.txt文件
    for i in range(len(all_frames_effect_keyPointInfo)):
        # print(all_frames_effect_keyPointInfo[i])
        [keyPointInfo, actionType] = all_frames_effect_keyPointInfo[i]
        # print(keyPointInfo, ' is ', actionType)
        # print(np.array(all_frames_effect_keyPointInfo).shape)  # (1215, 26)
        if keyPointInfo is not None:
            print(keyPointInfo)
            input()
            # write2Txt(fileDir, txtFile, keyPointInfo, actionType)
            # write2Txt(boxing_result_data, keyPointInfo, actionType)
        else:
            print('all_frames_effect_keyPointInfo保存的是所有非None personKeyPointInfo数据,不再会打印这行')
    endTime = time()  # 结束计时
    print('写结果用时%.3f' % (endTime - beginTime))
