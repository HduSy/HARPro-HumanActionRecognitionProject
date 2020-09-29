# xcopy E:\json\running-video\person01_running_d1_uncomp\*.* F:\XLDownload\dataSet\KTH\running\running-video\person01_running_d1_uncomp\ /s /e /c /y /h /r
import os

actions = ['walking', 'boxing', 'handclapping', 'handwaving', 'jogging', 'running']
action_type = 'walking'
dataSetDir = 'F:\\XLDownload\\dataSet\\KTH\\' + action_type + '\\' + action_type + '-video'
commands_file = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\src\\python-lrn\\command.txt'
command_list = []


# win bash批处理copy脚本 测一下然后copy
# pxdx===>personx_boxing_dx_uncomp
def generateCopyFiles():
    global command_list
    command_list = []
    for i in range(1, 26):
        for j in range(1, 5):
            # sourcePath = 'G:\\json\\person%d_' % i + action_type + '_d%d_uncomp' % j
            sourcePath = 'E:\\json\\' + action_type + '-video\\' + 'person%02d_' % i + action_type + '_d%d_uncomp' % j
            targetPath = 'F:\\XLDownload\\dataSet\\KTH\\' + action_type + '\\' + action_type + '-video\\person' + (
                    '%02d' % i) + '_' + action_type + '_d%d_uncomp\\' % j
            copyCommand = 'xcopy ' + sourcePath + '\\*.*' + ' ' + targetPath + ' ' + '/s /e /c /y /h /r'
            command_list.append(copyCommand)


def writeFile(fileName):
    global command_list
    if not os.path.exists(fileName):
        f = open(fileName, 'w')
    with open(fileName, 'w', encoding='utf-8') as f:
        for i in range(len(command_list)):
            f.write(command_list[i] + '\n')
        f.close()


if __name__ == '__main__':
    generateCopyFiles()
    writeFile(commands_file)
