# 55
txtPath = 'F:\\XLDownload\\dataSet\\KTH\\HARPro\\src\\paint\\data.txt'
result = []

AXS = []
AYS = []
AZS = []


def readTxt(filepath):
    g = 0
    with open(txtPath, 'r') as fileObj:
        lineNum = len(fileObj.readlines())
        g = lineNum // 28
    with open(txtPath, 'r') as fileObj:
        # lineNum = len(fileObj.readlines())
        i = 0
        while i < g:
            for k in range(3):
                fileObj.readline()
            for j in range(25):
                result.append(fileObj.readline())
            i += 1
    return result


readTxt(txtPath)
g1 = result[0:25]


def getAxs():
    i = 0
    while i < 1375:
        group = result[i:i + 25]
        ax = []
        ay = []
        az = []
        for j in range(25):
            strArr = group[j].split(' ')
            x = float(strArr[0])
            y = float(strArr[1])
            z = float(strArr[2])
            ax.append(x)
            ay.append(y)
            az.append(z)
        AXS.append(ax)
        AYS.append(ay)
        AZS.append(az)
        i += 25
        # print(i)


getAxs()
# print(AXS)
# print(AYS)
# print(AZS)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

for i in range(55):
    x = AXS[i]
    y = AYS[i]
    z = AZS[i]
    print(x, y, z)
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.scatter(x, y, z)

    # ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], c='black')
    # ax.plot([x[1], x[20]], [y[1], y[20]], [z[1], z[20]], c='black')
    # ax.plot([x[20], x[2]], [y[20], y[2]], [z[20], z[2]], c='black')
    # ax.plot([x[2], x[3]], [y[2], y[3]], [z[2], z[3]], c='black')
    ax.plot([x[1], x[2]], [y[1], y[2]], [z[1], z[2]], c='black')
    ax.plot([x[2], x[21]], [y[2], y[21]], [z[2], z[21]], c='black')
    ax.plot([x[21], x[3]], [y[21], y[3]], [z[21], z[3]], c='black')
    ax.plot([x[3], x[4]], [y[3], y[4]], [z[3], z[4]], c='black')


    # ax.plot([x[20], x[8]], [y[20], y[8]], [z[20], z[8]], c='black')
    # ax.plot([x[8], x[9]], [y[8], y[9]], [z[8], z[9]], c='black')
    # ax.plot([x[9], x[10]], [y[9], y[10]], [z[9], z[10]], c='black')
    # ax.plot([x[10], x[11]], [y[10], y[11]], [z[10], z[11]], c='black')
    # ax.plot([x[11], x[23]], [y[11], y[23]], [z[11], z[23]], c='black')
    # ax.plot([x[10], x[24]], [y[10], y[24]], [z[10], z[24]], c='black')
    #
    # ax.plot([x[20], x[4]], [y[20], y[4]], [z[20], z[4]], c='black')
    # ax.plot([x[4], x[5]], [y[4], y[5]], [z[4], z[5]], c='black')
    # ax.plot([x[5], x[6]], [y[5], y[6]], [z[5], z[6]], c='black')
    # ax.plot([x[6], x[7]], [y[6], y[7]], [z[6], z[7]], c='black')
    # ax.plot([x[6], x[22]], [y[6], y[22]], [z[6], z[22]], c='black')
    # ax.plot([x[7], x[21]], [y[7], y[21]], [z[7], z[21]], c='black')
    #
    # ax.plot([x[0], x[12]], [y[0], y[12]], [z[0], z[12]], c='black')
    # ax.plot([x[12], x[13]], [y[12], y[13]], [z[12], z[13]], c='black')
    # ax.plot([x[13], x[14]], [y[13], y[14]], [z[13], z[14]], c='black')
    # ax.plot([x[14], x[15]], [y[14], y[15]], [z[14], z[15]], c='black')
    #
    # ax.plot([x[0], x[16]], [y[0], y[16]], [z[0], z[16]], c='black')
    # ax.plot([x[16], x[17]], [y[16], y[17]], [z[16], z[17]], c='black')
    # ax.plot([x[17], x[18]], [y[17], y[18]], [z[17], z[18]], c='black')
    # ax.plot([x[18], x[19]], [y[18], y[19]], [z[18], z[19]], c='black')

    plt.savefig("F:\\XLDownload\\dataSet\\KTH\\HARPro\\src\\paint\\temp{}.png".format(i))  # 输入地址，并利用format函数修改图片名称
    plt.clf()  # 需要重新更新画布，否则会出现同一张画布上绘制多张图片
    # plt.close()
