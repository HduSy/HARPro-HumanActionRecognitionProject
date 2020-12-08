frames = []

data_path = 'F:\\XLDownload\\dataSet\\NTU-RGB-D\\s18-s32\\S027C002P011R001A080.skeleton'


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    global frames
    seq_info = read_skeleton(file)
    for index in range(len(seq_info['frameInfo'])):
        # 每一帧关节点dict list
        jointInfo = seq_info['frameInfo'][index]['bodyInfo'][0]['jointInfo']
        frame = []
        for i in range(len(jointInfo)):
            joints = [jointInfo[i]['x'], jointInfo[i]['y'], jointInfo[i]['z']]
            frame.extend([joints[0], joints[1], joints[2]])
        frame.append(file[-13:-9])
        frames.append(frame)
    return frames


if __name__ == '__main__':
    skeleton_data = read_xyz(data_path)
