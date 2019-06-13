# 环境
### 我运行程序用的是Matlab 2016b版本

# 目录结构
### \data---放的是.mat格式的数据
### \demo--demoXXX.m用于运行对应的算法，对应的手调参数请自行设置(邱博士完成)
### \demo4clone--参照demoXXX.m在clone数据集上运行对应的算法（张鸿辉完成）
### \evaluate_methods--评估各个迁移学习方法在不同ML上的效果
### \matlab2weka--matlab调用weka进行ML的封装，在matlab2weka基础上修改完成（张洪辉完成）
### \methods---放的是各个迁移学习方法的实现，其中包括的对应的paper（邱博士完成）
### \output---运行demoXXX.m后得到的结果（实时清空）
### \tools---一些工具方法
### \wekalab---matlab调用weka进行ML的封装，在wekalab基础上修改完成（张洪辉完成）
### \zhh_methods--评估各个迁移学习方法在不同ML上的效果

# demo文件
### XXX.m用于运行对应的算法，对应的手调参数请自行设置


# demo4clone文件夹
### 参照demoXXX.m，由张鸿辉完成，使用某一具体方法在clone数据集上进行的实验

# 目前版本包含的方法如下：
### DG
### NNFilter
### KMM
### TCA
### JDA
### JDM--该方法的运行请参考./methods/JDM下的How_to_run_JDM.md

