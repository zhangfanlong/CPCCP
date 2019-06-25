function [changingNames,creatingNames]=acquireNames(changingPath,creatingPath)
%导入changing文件下所有的mat文件
dirOutput = dir(fullfile(changingPath,'*.mat'));
changingNames = {dirOutput.name};

%导入creating文件下所有的mat文件
dirOutput = dir(fullfile(creatingPath,'*.mat'));
creatingNames = {dirOutput.name};

%将文件名后缀去掉
for i = 1:length(changingNames)
    temp = strsplit(changingNames{i},'.');
    changingNames{i} = temp{1};
end

%将文件名后缀去掉
for i = 1:length(creatingNames)
    temp = strsplit(creatingNames{i},'.');
    creatingNames{i} = temp{1};
end




end