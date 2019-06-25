%% 清空
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

%% 设置文件路径
changingPath = ['.',filesep,'data',filesep,'changing instance'];
creatingPath = ['.',filesep,'data',filesep,'creating instance'];

%获取所有文件的名字
[changingNames,creatingNames] = acquireNames(changingPath,creatingPath);

%% 调用不同的迁移学习方法进行评估
disp("Evaluate JDM methods on different classifiers:")
%JDM_evaluate(changingPath,creatingPath,changingNames,creatingNames);

disp("Evaluate KMM methods on different classifiers:")
%KMM_evaluate(changingPath,creatingPath,changingNames,creatingNames);

disp("Evaluate Filter methods on different classifiers:")
%Filter_evaluate(changingPath,creatingPath,changingNames,creatingNames);

disp("Evaluate DG methods on different classifiers:")
%DG_evaluate(changingPath,creatingPath,changingNames,creatingNames);

disp("Evaluate TCA methods on different classifiers:")
TCA_evaluate(changingPath,creatingPath,changingNames,creatingNames);

disp("Evaluate JDA methods on different classifiers:")
JDA_evaluate(changingPath,creatingPath,changingNames,creatingNames);

disp("All Evaluation are done!");
