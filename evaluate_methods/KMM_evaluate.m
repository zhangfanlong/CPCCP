function KMM_evaluate(changingPath,creatingPath,changingNames,creatingNames)

%importFile(changingPath,creatingPath)

% super-parameter
sigma=1;

% set result file
modelName = 'KMM';
file_name=['./output/',modelName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,sigma,dataset,target,source,f1,AUC';
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
                
                % call KMM
                alpha_weight = KMM('rbf',sourceX, targetX, sigma);
                alpha_weight = normalizeAlpha(alpha_weight, 1);
                
                % Logistic regression
                %model = train(alpha_weight, sourceY, sparse(sourceX), '-s 0 -c 1');
                %predictY = predict(targetY, sparse(targetX), model);
                %[accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(predictY, targetY);
                
                 %���û���ѧϰ��������
                learnerNames = {'RFC';'GPR';'SVM';'LR'};

                %ѭ������Ǩ��ѧϰ����and����ѧϰ����
                for index=1:4
                    %disp("===============================================in for" )
                    learnerName = learnerNames(index);
                    [f_measure,AUC,precision,recall,predictedY] = classifier_example(sourceX,sourceY,targetX,targetY,index);
                    %parameter string
                    resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC),',',num2str(precision),',',num2str(recall)]
                    disp(resultStr);
                    fprintf(file,'%s\n',resultStr);
                end              
            end
        end
    end
end
end