function Filter_evaluate(changingPath,creatingPath,changingNames,creatingNames)
% super-parameter
member = 10;

% set result file
learnerNames = {'RFC';'GPR';'SVM';'LR'};
modelName = 'Filter';
file_name=['./output/',modelName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dataset,target,source,R_average,P_average,F_average,AUC';
fprintf(file,'%s\n',headerStr);

% Select dataset
% 选择changing数据集和creating数据集
for dataset = [1,2]
    if dataset == 1
        fileList = changingNames;
        dataName = 'changing';
        %循环导入数据
        for q = 1:length(changingNames)
            %文件夹路径
            dirPath = changingPath;
            %文件名
            fileName = changingNames{q};
            %合并为文件的完整的绝对路径
            filePath = [dirPath,filesep,fileName,'.mat'];
            load(filePath);
        end
    elseif dataset == 2
        dataName = 'creating';
        fileList=creatingNames;
        %循环导入数据
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
                
                % call NNFilter
                [trainX, trainY] = NNFilter(member, sourceX, sourceY, targetX);
                
                
                %% Logistic regression
                learnerName = 'LR';
                model = train([], trainY, sparse(trainX), '-s 0 -c 1');
                predictY = predict(targetY, sparse(targetX), model);
                [~,~,~,precision,recall,f_measure,~,~, AUC] = evaluate_average(predictY, targetY);
                resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(recall),',',num2str(precision),',',num2str(f_measure),',',num2str(AUC)];
                fprintf(file,'%s\n',resultStr);
                disp([modelName,'_',learnerName,'_',dataName,' learning completed！'])
                %设置机器学习方法名字
                
               %% 循环调用迁移学习方法and机器学习方法
                %for index=1:4
                %    learnerName = learnerNames{index};
                %    [f_measure,AUC,precision,recall,~] = classifier_example(trainX,trainY,targetX,targetY,index);
                %    %parameter string
                %    resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(recall),',',num2str(precision),',',num2str(f_measure),',',num2str(AUC)];
                %    fprintf(file,'%s\n',resultStr);
                %    disp([modelName,'_',learnerName,'_',dataName,' learning completed！'])
                %end
            end
        end
    end
end
end
