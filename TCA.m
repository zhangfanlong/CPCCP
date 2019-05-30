% init program
clc();
clear();
addpath (genpath('.'))

% super-parameter
sigma=1;
dim=10;
mu=1;
lambda=1;

% set result file
learnerName = 'LR';
modelName = 'TCA';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dim,mu,lambda,dataset,target,source,f1,AUC';
fprintf(file,'%s\n',headerStr);

%�ļ�·��
%changingPath = 'D:\ѧϰ����\Ǩ��ѧϰ\Arffתmat\changing';
%creatingPath = 'D:\ѧϰ����\Ǩ��ѧϰ\Arffתmat\creating';
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
            %ע��windowa·��
            %filePath = [dirPath,'\',fileName,'.mat'];
            filePath = [dirPath,'/',fileName,'.mat'];
            %�����ļ�
            %ע����Ϊ�ڸ�ʽת��ʱ���������ֽ����˼򻯣��ԡ�-��Ϊ�и��������и��ҽ�������ǰ�沿��
            %���ļ���ArgoUML-resultsFeatureVector
            %��matlab�������еı�����Ϊ��ArgoUML
            load(filePath);
        end
        
        fileList= changingNames;
        attributeNum= length(changingNames) - 1;
        labelIndex=length(changingNames);
    elseif dataset == 2
        dataName = 'creating';
        fileList=creatingNames;
         %ѭ����������
        for q = 1:length(fileList)
            dirPath = creatingPath;
            fileName = fileList{q};
            filePath = [dirPath,'\',fileName,'.mat']
            
            load(filePath);
        end
        
        attributeNum=length(creatingNames) - 1;
        labelIndex=length(creatingNames);
    end
    
    % Select target project
    for i = 1:length(fileList)
        %���ļ�����׼����ȥ��-��������������ַ�����Ŀ����Ϊ�˶��ϵ���ı���
        temp = strsplit(fileList{i},'-');
        
        targetName = temp{1};
        
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
                model = train([], sourceY, sparse(newtrainX), '-s 0 -c 1');
                predictY = predict(targetY, sparse(newtestX), model);
                [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(predictY, targetY);
                
                %parameter string
                resultStr = [modelName,',',learnerName,',',num2str(dim),',',num2str(mu),',',num2str(lambda),',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC)]
                fprintf(file,'%s\n',resultStr);
            end
        end
    end
end