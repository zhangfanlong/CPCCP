%设置文件路径
%changingPath = 'D:\Code\Git\CPCCP\data\changing instance';
%creatingPath = 'D:\Code\Git\CPCCP\data\creating instance';
%%%Mac文件路径
changingPath = './data/changing instance';
creatingPath = './data/creating instance';

entrance(changingPath,creatingPath);


function addPath()
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
end

function start(changingPath,creatingPath,changingNames,creatingNames)
%调用不同的迁移学习方法进行评估
disp("Evaluate KMM methods on different classifiers:")
KMM_evaluate(changingPath,creatingPath,changingNames,creatingNames);
   
disp("Evaluate Filter methods on different classifiers:")
Filter_evaluate(changingPath,creatingPath,changingNames,creatingNames);
   
disp("Evaluate DG methods on different classifiers:")
DG_evaluate(changingPath,creatingPath,changingNames,creatingNames);
   
disp("Evaluate TCA methods on different classifiers:")
TCA_evaluate(changingPath,creatingPath,changingNames,creatingNames);
   
disp("Evaluate JDA methods on different classifiers:")
JDA_evaluate(changingPath,creatingPath,changingNames,creatingNames);
end

function entrance(changingPath,creatingPath)
addPath();
%初始化，导入数据集
[changingNames,creatingNames] = ImportData(changingPath,creatingPath);
%启动
start(changingPath,creatingPath,changingNames,creatingNames)
end