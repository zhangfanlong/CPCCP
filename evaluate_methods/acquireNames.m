function [changingNames,creatingNames]=acquireNames(changingPath,creatingPath)
%����changing�ļ������е�mat�ļ�
dirOutput = dir(fullfile(changingPath,'*.mat'));
changingNames = {dirOutput.name};

%����creating�ļ������е�mat�ļ�
dirOutput = dir(fullfile(creatingPath,'*.mat'));
creatingNames = {dirOutput.name};

%���ļ�����׺ȥ��
for i = 1:length(changingNames)
    temp = strsplit(changingNames{i},'.');
    changingNames{i} = temp{1};
end

%���ļ�����׺ȥ��
for i = 1:length(creatingNames)
    temp = strsplit(creatingNames{i},'.');
    creatingNames{i} = temp{1};
end




end