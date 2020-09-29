import os

# sudo cp -r ~/openpose-new/dataSet/KTH/walking/walking-video/person01_walking_d1_uncomp/result-json /media/lab607/HDU/json/walking-video/

actions = ['walking', 'boxing', 'handclapping', 'handwaving', 'jogging', 'running']
action_type = 'walking'
source = '~/openpose-new/dataSet/KTH/' + action_type + '/' + action_type + '-video/'
target = '/media/lab607/HDU/json/' + action_type + '-video/'
command_list = []
commands_file = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\src\\python-lrn\\command.txt'


def generateCopyCommand():
    for i in range(1, 26):
        for j in range(1, 5):
            dirName = 'person' + ('%02d' % i) + '_' + action_type + '_d' + ('%d' % j) + '_uncomp/'
            # print(dirName)
            copyCommand = 'sudo cp -r ' + source + dirName + 'result-json ' + target + dirName
            # copyCommand += '\necho %0d' % i
            command_list.append(copyCommand)
    # print(command_list)


def writeFile(fileName):
    global command_list
    if not os.path.exists(fileName):
        f = open(fileName, 'w')
    with open(fileName, 'w', encoding='utf-8') as f:
        for i in range(len(command_list)):
            f.write(command_list[i] + '\n')
        f.close()


if __name__ == '__main__':
    generateCopyCommand()
    writeFile(commands_file)
