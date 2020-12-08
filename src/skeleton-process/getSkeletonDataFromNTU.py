skeletionFilePath = 'F:\\Program Files (x86)\\PycharmProjects\\HARPro-HumanActionRecognitionProject\\src\\skeleton-process\\S027C002P011R001A080.skeleton'


def readSkeletonData(filePath):
    with open(filePath, 'r') as f:
        numFrame = int(f.readline())
        frameInfo = []
        for i in range(numFrame):
            numBody = int(f.readline())  # 1
            for b in range(numBody):
                f.readline()  # 舍弃一行
                numJoint = int(f.readline())  # 25
                # [[{x:,y:,z:},{},{}...{}], {'action': action_type}]
                jointInfo = []
                for v in range(numJoint):
                    line = f.readline().split()
                    x = float(line[0])
                    y = float(line[1])
                    z = float(line[2])
                    jointInfo.append({'x': x, 'y': y, 'z': z})
                sampleLabel = {'action': filePath[-13:-9]}
            frameInfo.append([jointInfo, sampleLabel])
            # print(frameInfo)
        return frameInfo


if __name__ == '__main__':
    skeletonData = readSkeletonData(skeletionFilePath)
    print(len(skeletonData))
