%���
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

%�����ļ�·��
changingPath = 'D:\ѧϰ����\Ǩ��ѧϰ\CPDP-master\data\changing';
creatingPath = 'D:\ѧϰ����\Ǩ��ѧϰ\CPDP-master\data\creating';

%��ʼ�����������ݼ�
[changingNames,creatingNames] = matImport(changingPath,creatingPath);

%���û���ѧϰ��������
learnerNames = {'RFC';'GPR';'SVM';'LR'};

%ѭ������Ǩ��ѧϰ����and����ѧϰ����
for choice=1:4
   learnerName = learnerNames{choice};
   KMM_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   Filter_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   DG_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   TCA_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
   JDA_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
end
   