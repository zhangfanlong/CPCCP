%% ���
clc();
clear();
%���·��
addpath (genpath('.'))
addpath([pwd filesep 'matlab2weka']);
if strcmp(filesep, '\')% Windows
    javaaddpath('D:\Software\Weka3.6.12\Weka-3-6\weka.jar');
elseif strcmp(filesep, '/')% Mac OS X
    javaaddpath('/Applications/weka-3-6-12/weka.jar')
end
javaaddpath([pwd filesep 'matlab2weka' filesep 'matlab2weka' filesep 'matlab2weka.jar']);

%% �����ļ�·��
changingPath = ['.',filesep,'data',filesep,'changing instance'];
creatingPath = ['.',filesep,'data',filesep,'creating instance'];

%��ȡ�����ļ�������
[changingNames,creatingNames] = acquireNames(changingPath,creatingPath);

%% ���ò�ͬ��Ǩ��ѧϰ������������
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
