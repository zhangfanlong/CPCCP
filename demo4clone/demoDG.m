% init program
clc();
clear();
addpath (genpath('.'))
addpath([pwd filesep 'matlab2weka']);

if strcmp(filesep, '\')% Windows    
    javaaddpath('D:\Software\Weka3.6.12\Weka-3-6\weka.jar');
elseif strcmp(filesep, '/')% Mac OS X
    javaaddpath('/Applications/weka-3-6-12/weka.jar')
end

javaaddpath([pwd filesep 'matlab2weka' filesep 'matlab2weka' filesep 'matlab2weka.jar']);

% set result file
learnerName = 'GPR';
modelName = 'DG';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dataset,target,source,f1,AUC';
fprintf(file,'%s\n',headerStr);


%文件路径
changingPath = 'D:\学习资料\迁移学习\Arff转mat\changing';
creatingPath = 'D:\学习资料\迁移学习\Arff转mat\creating';

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

% Select dataset
% 选择changing数据集和creating数据集
for dataset = [1,2]
    if dataset == 1
        fileList = changingNames;
        dataName = 'changing';
        %循环导入数据          
        for q = 1:length(fileList)
            %文件夹路径
            dirPath = changingPath;
            %文件名
            fileName = fileList{q};
            %合并为文件的完整的绝对路径
            filePath = [dirPath,'\',fileName,'.mat'];
            %导入文件
            %注：因为在格式转换时将变量名字进行了简化，以‘-’为切割点进行了切割且仅保留了前面部分
            %如文件：ArgoUML-resultsFeatureVector
            %在matlab工作区中的变量名为：ArgoUML
            load(filePath);
        end
        
        fileList= changingNames;
        
    elseif dataset == 2
        dataName = 'creating';
        fileList=creatingNames;
         %循环导入数据
        for q = 1:length(fileList)
            dirPath = creatingPath;
            fileName = fileList{q};
            filePath = [dirPath,'\',fileName,'.mat']
            
            load(filePath);
        end

    end
    
    % Select target project
    for i = 1:length(fileList)
        %将文件名标准化，去掉-（包括）后面的字符串，目的是为了对上导入的变量
        temp = strsplit(fileList{i},'-');
        targetName = temp{1};
        attributeNum = size(eval(targetName),2) - 1;
        labelIndex = size(eval(targetName),2);
        
        targetData=eval(targetName);
        targetData(targetData(:,labelIndex)==-1,labelIndex)=0;
        targetX = targetData(:,1:attributeNum);
        targetX = zscore(targetX);
        targetY = targetData(:,labelIndex);
        
        % Select source project
        for j = 1:length(fileList)
            temp = strsplit(fileList{j},'-');
            sourceName = temp{1};
            attributeNum = size(eval(sourceName),2) - 1;
            labelIndex = size(eval(sourceName),2);
            if(i~=j)
                sourceData=eval(sourceName);
                sourceData(sourceData(:,labelIndex)==-1,labelIndex)=0;
                sourceX = sourceData(:,1:attributeNum);
                sourceX = zscore(sourceX);
                sourceY = sourceData(:,labelIndex);
                
                % call DG
                alpha_weight = cal_data_gravitation(targetX, sourceX);
%                 alpha_weight = NormalizeAlpha(alpha_weight, 1);
                
                % Logistic regression
                %model = train(alpha_weight, sourceY, sparse(sourceX), '-s 0 -c 1');
                %predictY = predict(targetY, sparse(targetX), model);
                %[accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(predictY, targetY);
                
                %Support vector machine
                [f_measure,AUC,predictedY] = classifier_example(sourceX,sourceY,targetX,targetY);
                
                %parameter string
                resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC)]
                fprintf(file,'%s\n',resultStr);
            end
        end
    end
end