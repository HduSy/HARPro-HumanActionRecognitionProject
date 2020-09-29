import os

actions = ['walking', 'boxing', 'handclapping', 'handwaving', 'jogging', 'running']
action_type = 'running'
video_file_base_dir = 'F:\\XLDownload\\dataSet\\KTH\\' + action_type + '\\'
# --write_images=/home/lab607/openpose-new/dataSet/KTH/walking/walking-video/%s/result-images
walking_command = './openpose.bin --video=/home/lab607/openpose-new/dataSet/KTH/walking/walking-video/%s --display=0 --render_pose=2  --write_json=/home/lab607/openpose-new/dataSet/KTH/walking/walking-video/%s/result-json'
# --write_images=/home/lab607/openpose-new/dataSet/KTH/' + action_type + '/' + action_type + '-video/%s/result-images
action_command = './openpose.bin --video=/home/lab607/openpose-new/dataSet/KTH/' + action_type + '/' + action_type + '-video/%s --display=0 --render_pose=2  --write_json=/home/lab607/openpose-new/dataSet/KTH/' + action_type + '/' + action_type + '-video/%s/result-json'
command_list = []
commands_file = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\src\\python-lrn\\command.txt'


# 遍历数据集某类动作文件夹下所有视频文件名并生成相应命令
def generateCommand(file_dir):
    global command_list
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
    # print(files)  # ['person01_running_d1_uncomp.avi', 'person01_running_d2_uncomp.avi',...]
    for i in range(len(files)):
        # print(files[i])
        # print(walking_command % (files[i]))
        file_dir = files[i].split('.')[0]
        command = action_command % (files[i], file_dir)
        createDir = 'sudo mkdir /home/lab607/openpose-new/dataSet/KTH/' + action_type + '/' + action_type + '-video/%s' % (
            file_dir)
        command_list.append(createDir + '\n' + command)
    # return command_list


def writeFile(fileName):
    global command_list
    if not os.path.exists(fileName):
        f = open(fileName, 'w')
    with open(fileName, 'w', encoding='utf-8') as f:
        for i in range(len(command_list)):
            f.write(command_list[i] + '\n')
        f.close()


# def openFile(fileName):


if __name__ == '__main__':
    generateCommand(video_file_base_dir)
    writeFile(commands_file)
