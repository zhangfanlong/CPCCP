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
            filePath = [dirPath,filesep,fileName,'.mat'];
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
                
                 %设置机器学习方法名字
                learnerNames = {'RFC';'GPR';'SVM';'LR'};

                %循环调用迁移学习方法and机器学习方法
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