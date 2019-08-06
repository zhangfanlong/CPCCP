%% ���
clc();
clear();
%���·��
addpath (genpath('.'))  
addpath([pwd filesep 'matlab2weka']);

javaaddpath([pwd filesep 'matlab2weka' filesep 'matlab2weka' filesep 'matlab2weka' filesep 'lib' filesep 'weka.jar'])
javaaddpath([pwd filesep 'matlab2weka' filesep 'matlab2weka' filesep 'matlab2weka.jar']);

%% �����ļ�·��
changingPath = ['.',filesep,'data',filesep,'changing instance'];
creatingPath = ['.',filesep,'data',filesep,'creating instance'];

%��ȡ�����ļ�������
[changingNames,creatingNames] = acquireNames(changingPath,creatingPath);

%% ���ò�ͬ��Ǩ��ѧϰ������������
disp("Evaluate JDM methods on different classifiers:")
%JDM_evaluate(changingPath,creatingPath,changingNames,creatingNames);

disp("Evaluate KMM methods on different classifies:")
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
