function TCA_example(learnerName,choice,changingPath,creatingPath,changingNames,creatingNames)
% super-parameter
sigma=1;
dim=10;
mu=1;
lambda=1;

% set result file
modelName = 'TCA';
file_name=['./output/',modelName,'_',learnerName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dim,mu,lambda,dataset,target,source,f1,AUC';
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
            %将文件名标准化，去掉-（包括）后面的字符串，目的是为了对上导入的变量
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
                [f_measure,AUC,predictedY] = classifier_example(newtrainX,sourceY,newtestX,targetY,choice);
                
                %parameter string
                resultStr = [modelName,',',learnerName,',',num2str(dim),',',num2str(mu),',',num2str(lambda),',',dataName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC)]
                fprintf(file,'%s\n',resultStr);
            end
        end
    end
end
end