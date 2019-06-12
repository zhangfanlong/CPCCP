function Filter_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
% super-parameter
member = 10;

% set result file

modelName = 'Filter';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,member,dataset,target,source,f1,AUC';
fprintf(file,'%s\n',headerStr);

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
            filePath = [dirPath,'\',fileName,'.mat'];
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
            filePath = [dirPath,'\',fileName,'.mat']
            
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
                
                % call NNFilter
                [trainX, trainY] = NNFilter(member, sourceX, sourceY, targetX);
                
                % Logistic regression
                %model = train([], trainY, sparse(trainX), '-s 0 -c 1');
                %predictY = predict(targetY, sparse(targetX), model);
                %[accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(predictY, targetY);
                
                %Support vector machine
                [f_measure,AUC,precision,recall,predictedY] = classifier_example(sourceX,sourceY,targetX,targetY,choice);
                
                %parameter string
                resultStr = [modelName,',',learnerName,',',num2str(member),',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(precision),',',num2str(recall)]
                fprintf(file,'%s\n',resultStr);
            end
        end
    end
end
end
