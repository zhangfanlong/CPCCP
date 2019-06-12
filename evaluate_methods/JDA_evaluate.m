function TCA_evaluate(changingPath,creatingPath,changingNames,creatingNames)
    % super-parameter
    sigma=1;
    dim=10;
    lambda=1;

    % set result file
    learnerNames = {'RFC';'GPR';'SVM';'LR'};
    modelName = 'JDA';
    file_name=['./output/',modelName,'_result.csv'];
    file=fopen(file_name,'w');
    headerStr = 'model,learner,dataset,target,source,P_average,R_average,F_average,AUC';
    fprintf(file,'%s\n',headerStr);

    % JDA process file
    %process_f1_file_name = ['./output/JDA_process_f1_result.csv'];
    %process_f1_file=fopen(process_f1_file_name,'w');
    %process_AUC_file_name = ['./output/JDA_process_AUC_result.csv'];
    %process_AUC_file=fopen(process_AUC_file_name,'w');

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
    
    % Select tar ·get project
    for i = 1:length(fileList)
        %将文件名标准化，去掉-（包括）后面的字符串，目的是为了对上导入的变量
        temp = strsplit(fileList{i},'-'); 
        targetName = temp{1};
        targetData=eval(targetName);
        attributeNum = size(eval(targetName),2) - 1;
        labelIndex = size(eval(targetName),2);
        
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
                
                % call JDA
                JDA_options.k = dim;
                JDA_options.lambda = lambda;
                JDA_options.ker = 'linear';            % 'primal' | 'linear' | 'rbf'
                JDA_options.gamma = sigma          % kernel bandwidth: rbf only
                
                % init pseudo-label 
                %LRModel = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
                %LR_Cls = predict(targetY, sparse(targetX), LRModel);
                %[accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(LR_Cls, targetY);
                
                %设置机器学习方法名字
                

                %循环调用迁移学习方法and机器学习方法
                for index=1:4
                    learnerName = learnerNames{index};
                    [f_measure,AUC,precision,recall,predictedY] = classifier_example(sourceX,sourceY,targetX,targetY,index);
                    %parameter string
                    resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(recall),',',num2str(precision),',',num2str(f_measure),',',num2str(AUC)];
                    fprintf(file,'%s\n',resultStr);
                    disp([modelName,'_',learnerName,' learning completed！'])
                end
                
                %LRClsArray = [];
                %LRClsArray = [LRClsArray,LR_Cls];
                %fprintf(process_f1_file,'%f,',f_measure);
                %fprintf(process_AUC_file,'%f,',AUC);
                
                %for t = 1:10
                %    [newtrainX, newtestX] = JDA(sourceX,targetX,sourceY,LR_Cls,JDA_options);
                %    LRModel = train([], sourceX, sparse(newtrainX), '-s 0 -c 1');
                %    LR_Cls = predict(targetY, sparse(newtestX), LRModel);
                    
                %    [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC,AUC] = evaluate(LR_Cls, targetY);
                %    fprintf(process_f1_file,'%f,',f_measure);
                %    fprintf(process_AUC_file,'%f,',AUC);
                %end
                
                %fprintf(process_f1_file,'\n');
                %fprintf(process_AUC_file,'\n');
                
           end
            
        end
    end
end
