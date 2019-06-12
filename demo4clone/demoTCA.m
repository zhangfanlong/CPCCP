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


% super-parameter
sigma=1;
dim=10;
mu=1;
lambda=1;

% set result file
learnerName = 'GPR';
modelName = 'TCA';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dim,mu,lambda,dataset,target,source,f1,AUC';
fprintf(file,'%s\n',headerStr);

%�ļ�·��
%%%windows�ļ�·��
%changingPath = 'D:\ѧϰ����\Ǩ��ѧϰ\Arffתmat\changing';
%creatingPath = 'D:\ѧϰ����\Ǩ��ѧϰ\Arffתmat\creating';
%%%Mac�ļ�·��
changingPath = './data/changing instance';
creatingPath = './data/creating instance';

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

% Select dataset
% ѡ��changing���ݼ���creating���ݼ�
for dataset = [1,2]
    if dataset == 1
        fileList = changingNames;
        dataName = 'changing';
        %ѭ����������          
        for q = 1:length(fileList)
            %�ļ���·��
            dirPath = changingPath;
            %�ļ���
            fileName = fileList{q};
            %�ϲ�Ϊ�ļ��������ľ���·��
            filePath = [dirPath,filesep,fileName,'.mat'];
            %�����ļ�
            %ע����Ϊ�ڸ�ʽת��ʱ���������ֽ����˼򻯣��ԡ�-��Ϊ�и��������и��ҽ�������ǰ�沿��
            %���ļ���ArgoUML-resultsFeatureVector
            %��matlab�������еı�����Ϊ��ArgoUML
            load(filePath);
        end
        
        fileList= changingNames;
    elseif dataset == 2
        dataName = 'creating';
        fileList=creatingNames;
         %ѭ����������
        for q = 1:length(fileList)
            dirPath = creatingPath;
            fileName = fileList{q};
            filePath = [dirPath,filesep,fileName,'.mat']
            
            load(filePath);
        end
    end
    
    % Select target project
    for i = 1:length(fileList)
        %���ļ�����׼����ȥ��-��������������ַ�����Ŀ����Ϊ�˶��ϵ���ı���
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
            %���ļ�����׼����ȥ��-��������������ַ�����Ŀ����Ϊ�˶��ϵ���ı���
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
                
                % call TCA
                options = tca_options('Kernel', 'linear', 'KernelParam', sigma, 'Mu', mu, 'lambda', lambda, 'Dim', dim);
                [newtrainX, ~, newtestX] = tca(sourceX, targetX, targetX, options);
                
                % Logistic regression
                %model = train([], sourceY, sparse(newtrainX), '-s 0 -c 1');
                %predictY = predict(targetY, sparse(newtestX), model);
                %[accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(predictY, targetY);
                
                %Support vector machine
                [f_measure,AUC,predictedY] = classifier_example(newtrainX,sourceY,newtestX,targetY);
                
                %parameter string
                resultStr = [modelName,',',learnerName,',',num2str(dim),',',num2str(mu),',',num2str(lambda),',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC)]
                fprintf(file,'%s\n',resultStr);
            end
        end
    end
end