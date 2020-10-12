## 基于姿态序列的人体行为识别-papercode

### Some Tutorials
[莫烦Python](https://mofanpy.com/)

[keras配置教程](https://www.pythonf.cn/read/123617)

[tensorflow缺少tutials文件夹 解决](https://www.cnblogs.com/tszr/p/12060124.html)

[tensorflow下载速度巨慢解决方案-迅雷下载.whl文件,然后pip install](https://blog.csdn.net/qq_39234705/article/details/83241129)

[tensorflow 2.1 import ERROR DLL load failed-升级python到3.7安装](https://www.cnblogs.com/bjxqmy/p/12661931.html)

- [卷积层](https://mofanpy.com/tutorials/machine-learning/keras/intro-CNN/#%E5%8D%B7%E7%A7%AF%20%E5%92%8C%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
- [池化层](https://mofanpy.com/tutorials/machine-learning/keras/intro-CNN/#%E6%B1%A0%E5%8C%96(pooling))
- [全连接层](https://www.cnblogs.com/Terrypython/p/11147665.html)
- [Dropout](https://zhuanlan.zhihu.com/p/38200980)
- [注意力机制](https://blog.csdn.net/uhauha2929/article/details/80733255)
- [Understanding-LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [sktlearn-MinMaxScaler-归一化方法公式:](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) ![归一化完整公式](https://private.codecogs.com/gif.latex?X_scaled%20%3D%20%5Cfrac%7B%20%28X%20-%20X.min%28axis%3D0%29%29%20%7D%7B%20%28X.max%28axis%3D0%29%20-%20X.min%28axis%3D0%29%29%7D%20%5Ccdot%20%28max%20-%20min%29&plus;min)
- [模型过拟合问题](https://blog.csdn.net/weixin_43593330/article/details/103799225)
- [特征融合](http://html.rhhz.net/buptjournal/html/20170401.htm)
- [保存自定义层加载模型文件错误时解决方案](https://blog.csdn.net/program_developer/article/details/90453946)

### Promblem Resolved
- 数据预处理:有效关键点个数\排除多目标干扰\多目标选择 raw data->sample data->feature data->net design->train & test
- 稀疏采样
- 归一化[0,1]解决Spatial Feature尺度归一避免传统图像识别领域SIFT复杂化方法
- 归一化[-1,1]解决Tempral Angle Feature? No!角度特征归一化无意义
- Temperal Feature特征采用角度特征 来避免d2类视频中缩放造成的尺度变化对距离特征的影响-准确率有所提高3-5个百分点
  1. (1,8)<===>(2,3)
  2. (1,8)<===>(5,6)
  3. (1,8)<===>(3,4)
  4. (1,8)<===>(6,7)
  5. (2,3)<===>(9,10)
  6. (2,3)<===>(10,11)
  7. (3,4)<===>(9,10)
  8. (3,4)<===>(10,11)
  9. (5,6)<===>(12,13)
  10. (5,6)<===>(13,14)
  11. (6,7)<===>(12,13)
  12. (6,7)<===>(13,14)
  13. (3,2)<===>(3,4)
  14. (6,5)<===>(6,7)
  15. (8,1)<===>(9,10)
  16. (8,1)<===>(12,13)
  17. (10,9)<===>(10,11)
  18. (13,12)<===>(13,14)
  19. (11,10)<===>(11,12)
  20. (14,13)<===>(14,19)
  21. (2,3)<===>(5,6)
  22. (3,4)<===>(6,7)
  23. (9,10)<===>(12,13)
  24. (10,11)<===>(13,14)
  25. (11,22)<===>(14,19)
- 参数过多导致过拟合问题 dropout=0.7效果最佳
- 特征融合(特征级融合属于先映射融合)
- 添加注意力机制
- 网络模型设计
