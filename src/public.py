# 全局变量
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking', 'falling']
txtDir = 'F:\\Program Files (x86)\\PycharmProjects\\HARPro-HumanActionRecognitionProject\\action'
action_type = 'boxing'
model_filename = 'F:\\Program Files (x86)\\PycharmProjects\\HARPro-HumanActionRecognitionProject\\src\\model-file\\HAR.h5'
# 数据集配置
regularization = True  # 归一化与否
anglization = True  # 时序角度与否
# 网络配置
spatial_attention = True  # 空间注意力
temporal_attention = True  # 时间注意力
