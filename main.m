%清空
clc();
clear();

%添加路径
addpath (genpath('.'))
addpath([pwd filesep 'matlab2weka']);
if strcmp(filesep, '\')% Windows    
    javaaddpath('D:\Software\Weka3.6.12\Weka-3-6\weka.jar');
elseif strcmp(filesep, '/')% Mac OS X
    javaaddpath('/Applications/weka-3-6-12/weka.jar')
end
javaaddpath([pwd filesep 'matlab2weka' filesep 'matlab2weka' filesep 'matlab2weka.jar']);

%设置文件路径
changingPath = 'D:\学习资料\迁移学习\CPDP-master\data\changing';
creatingPath = 'D:\学习资料\迁移学习\CPDP-master\data\creating';

%初始化，导入数据集
[changingNames,creatingNames] = matImport(changingPath,creatingPath);

%设置机器学习方法名字
learnerNames = {'RFC';'GPR';'SVM';'LR'};

%循环调用迁移学习方法and机器学习方法
for choice=1:4
   learnerName = learnerNames{choice};
   KMM_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   Filter_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   DG_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   TCA_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   JDA_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
end
   