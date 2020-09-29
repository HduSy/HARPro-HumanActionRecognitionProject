import os

actions = ['walking', 'boxing', 'handclapping', 'handwaving', 'jogging', 'running']
action_type = 'walking'
dataSetDir = 'F:\\XLDownload\\dataSet\\KTH\\' + action_type + '\\' + action_type + '-video'
# dataSetDir = 'E:\\json\\' + action_type + '-video'


# 生成文件夹 personx_actionType_dx_uncomp
def generateDirs():
    dirsNames = []
    for i in range(1, 26):
        for j in range(1, 5):
            dirName = 'person' + ('%02d' % i) + '_' + action_type + '_d' + ('%d' % j) + '_uncomp'
            # print(dirName)
            dirsNames.append(dirName)
    for i in range(len(dirsNames)):
        if not os.path.exists(dataSetDir):
            os.mkdir(dataSetDir)
        if not os.path.exists(dataSetDir + '\\' + dirsNames[i]):
            os.mkdir(dataSetDir + '\\' + dirsNames[i])
        if not os.path.exists(dataSetDir + '\\' + dirsNames[i] + '\\result-json'):
            os.mkdir(dataSetDir + '\\' + dirsNames[i] + '\\result-json')


if __name__ == '__main__':
    generateDirs()
