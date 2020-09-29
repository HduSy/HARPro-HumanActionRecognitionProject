import os
from time import time

actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
action_type = 'boxing'
txtDir = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\action'
dataSet = []


# test case:86 18 0.809911 94 28 0.658329 95 28 0.484622 82 38 0.429741 72 38 0.576679 92 27 0.709032 81 35 0.734206 71 38 0.776877 92 57 0.545291 93 57 0.378327 92 81 0.474657 92 100 0.481058 92 58 0.595076 92 81 0.593236 94 102 0.647892 86 16 0.171122 87 17 0.82526 0 0 0.0 93 18 0.796703 86 105 0.621151 88 106 0.596866 95 105 0.561604 86 101 0.419428 86 100 0.369899 95 102 0.114403 boxing
def transformTxtLine2ListObj(line):
    tmp = []
    res = []
    lineArr = line.split(' ')
    for index in range(25):
        x = int(lineArr[3 * index + 0])
        y = int(lineArr[3 * index + 1])
        score = float(lineArr[3 * index + 2])
        tmp.append({'x': x, 'y': y, 'score': score})
    type = lineArr[75]
    # TODO:tmp结构按需调整
    # tmp.append(type)
    res.extend([tmp, {'action': type}])
    # print(tmp)
    # print(res)
    return res


def readDataFromTxt(filePath, all=False, actionType=None):
    global txtDir
    global actions
    if all:
        for index in range(6):
            filePath += '\\' + actions[index] + '\\' + actions[index] + '-result-data.txt'
            with open(filePath, 'r') as fileObj:
                line = fileObj.readline()
                while line:
                    # print(line, end='')
                    dataSet.append(line.strip('\n'))
                    line = fileObj.readline()
                fileObj.close()
            filePath = txtDir
    else:
        filePath += '\\%s\\%s-result-data.txt' % (actionType, actionType)
        with open(filePath, 'r') as fileObj:
            line = fileObj.readline()
            while line:
                # print(line, end='')
                dataSet.append(transformTxtLine2ListObj(line.strip('\n')))
                line = fileObj.readline()
            fileObj.close()
        filePath = txtDir
        print('仅仅读取%s动作' % actionType)
    return dataSet


if __name__ == '__main__':
    # global actions
    beginTime = time()
    # readDataFromTxt(txtDir, True, None)
    readDataFromTxt(txtDir, False, actions[0])
    endTime = time()
    print('耗时:%f' % (endTime - beginTime))
