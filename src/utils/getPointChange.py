import os
import json
import numpy as np
import xlwt

from src.refineAndSaveKeyPointData import readJSON, getKeyPointInfo2

video = '00047'
dataSetDir = 'F:\\video\\' + video
dirName = dataSetDir + '\\' + 'result-json'
p38 = []


def write2Excel(t, val):
    f = xlwt.Workbook()  # 创建工作薄
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    sheet1.write(0, 0, 't')
    sheet1.write(0, 1, 'val')
    it = len(val)
    j = 1
    for i in t:
        sheet1.write(j, 0, i)  # 循环写入
        j = j + 1
    j = 1
    for i in val:
        sheet1.write(j, 1, i)  # 循环写入
        j = j + 1
    f.save('text.xls')  # 保存文件


if __name__ == '__main__':
    for root, dirs, files in os.walk(dataSetDir):
        jsonFiles = files
    for i in range(len(jsonFiles)):
        # print(dirName + '\\' + jsonFiles[i])
        personKeyPointInfo = getKeyPointInfo2(dirName + '\\' + jsonFiles[i])
        # print(personKeyPointInfo[0])
        personKeyPointList = personKeyPointInfo[0]
        MidHip = np.array([personKeyPointList[8]['x'], personKeyPointList[8]['y']])
        keyPointArray = np.array([personKeyPointList[3]['x'], personKeyPointList[3]['y']])
        EDistance = np.sqrt(np.sum(np.square(keyPointArray - MidHip)))
        p38.append(EDistance)
    t = [i for i in range(len(p38))]
    write2Excel(t, p38)
