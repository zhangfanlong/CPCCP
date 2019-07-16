function KMM_evaluate(changingPath,creatingPath,changingNames,creatingNames)

%importFile(changingPath,creatingPath)

% super-parameter
sigma=1;

% set result file
learnerNames = {'RFC';'GPR';'SVM';'LR'};
modelName = 'KMM';
file_name=['./output/',modelName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dataset,target,source,R_average,P_average,F_average,AUC';
fprintf(file,'%s\n',headerStr);

% Select dataset
% ѡ��changing���ݼ���creating���ݼ�
for dataset = [1,2]
    if dataset == 1
        fileList = changingNames;
        dataName = 'changing';
        %ѭ����������
        for q = 1:length(changingNames)
            %�ļ���·��
            dirPath = changingPath;
            %�ļ���
            fileName = changingNames{q};
            %�ϲ�Ϊ�ļ��������ľ���·��
            filePath = [dirPath,filesep,fileName,'.mat'];
            load(filePath);
        end
    elseif dataset == 2
        dataName = 'creating';
        fileList=creatingNames;
        %ѭ����������
        for q = 1:length(creatingNames)
            dirPath = creatingPath;
            fileName = creatingNames{q};
            filePath = [dirPath,filesep,fileName,'.mat']
            load(filePath);
        end
    end
    % Select target project
    for i = 1:length(fileList)
        targetName = fileList{i};
        attributeNum = size(eval(targetName),2) - 1;
        labelIndex = size(eval(targetName),2);
        
        targetData=eval(targetName);
        targetData(targetData(:,labelIndex)==-1,labelIndex)=0;
        targetX = targetData(:,1:attributeNum);
        targetX = zscore(targetX);
        targetY = targetData(:,labelIndex);
        
        % Select source project
        for j = 1:length(fileList)
            sourceName = fileList{j};
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
                
                %% Logistic regression
                learnerName = 'LR';
                model = train(alpha_weight, sourceY, sparse(sourceX), '-s 0 -c 1');
                predictY = predict(targetY, sparse(targetX), model);
                [~,~,~,precision,recall,f_measure,~,~, AUC] = evaluate_average(predictY, targetY);
                resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(recall),',',num2str(precision),',',num2str(f_measure),',',num2str(AUC)];
                fprintf(file,'%s\n',resultStr);
                disp([modelName,'_',learnerName,'_',dataName,' learning completed��'])
                
                %���û���ѧϰ��������
                
                
                %% ѭ������Ǩ��ѧϰ����and����ѧϰ����
                %for index=1:4
                %    learnerName = learnerNames{index};
                %    [f_measure,AUC,precision,recall,~] = classifier_example(sourceX,sourceY,targetX,targetY,index);
                %    %parameter string
                %    resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(recall),',',num2str(precision),',',num2str(f_measure),',',num2str(AUC)];
                %    fprintf(file,'%s\n',resultStr);
                %    disp([modelName,'_',learnerName,'_',dataName,' learning completed��'])
                %end
            end
        end
    end
end
end