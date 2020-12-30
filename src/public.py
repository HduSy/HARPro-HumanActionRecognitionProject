# 全局变量
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking', 'falling']
txtDir = 'F:\\Program Files (x86)\\PycharmProjects\\clone\\HARPro-HumanActionRecognitionProject\\action'
action_type = 'falling'
# model_filename = 'F:\\Program Files (x86)\\PycharmProjects\\HARPro-HumanActionRecognitionProject\\src\\model-file\\HAR.h5'
model_filename = 'F:\\Program Files (x86)\\PycharmProjects\\clone\\HARPro-HumanActionRecognitionProject\\src\\checkpoints\\model_best.h5'
# 数据集配置
regularization = True  # 归一化与否
anglization = True  # 时序角度与否
# 网络配置
spatial_attention = True  # 空间注意力
temporal_attention = True  # 时间注意力
# 骨架序列长度
frameNum = 24  # n_input time_step步长 TODO:可以做下对比实验啊
# 时空特征维度
inputA_dim = 14
inputBA_dim = 8
inputBL_dim = 7
if anglization:
    inputB_dim = inputBA_dim
else:
    inputB_dim = inputBL_dim
