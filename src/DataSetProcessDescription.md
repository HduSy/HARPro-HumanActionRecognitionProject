## 数据集处理流程

- 修改`generateExtractJsonUbuntuCommand.py`参数生成openpose提取关键点Ubuntu命令脚本.sh,注意不必保存images,只需保存json数据;
- Ubuntu运行脚本提取关键点
- 修改`generateWinDirCommand.py`路径参数在win和U上生成对应文件夹
- Ubuntu2U(确保U盘已插):执行`generateCopyUbuntu2UCommand.py`生成ubuntu2u拷贝命令,修改并上传至Ubuntu`CopyUbuntu2UCommand.sh`需要进行一次`fromdos CopyUbuntu2UCommand.sh`
- U2Win(确保U盘已插):执行`generateCopyU2WinCommand.py`生成u2win拷贝命令,修改并执行`cpU2W.bat`
- 执行`refineAndSaveKeyPointData.py`,调用`write2Txt`函数,迭代处理关键点文件夹中json数据,存入`.txt`文件
- `generateSTFeatures.py`后续处理...可以先一步保存中间结果或直接提取关键点

## 问题解决
- 解决windows系统制作.sh脚本在Ubuntu上运行报错 '$''\r' Invalid Argument 问题 ```sudo apt-get install tofrodos``` [tofrodos](https://www.jianshu.com/p/d5eb279de997)