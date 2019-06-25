function JDM_evaluate(changingPath,creatingPath,changingNames,creatingNames)
% set result file
learnerNames = {'RFC';'GPR';'SVM';'LR'};
modelName = 'JDM';
file_name=['./output/',modelName,'_result.csv'];
file=fopen(file_name,'w');
headerStr = 'model,learner,dataset,target,source,P_average,R_average,F_average,AUC';
fprintf(file,'%s\n',headerStr);

% Set parameter
sigma = 0.1; % Function width of gaussian kernel
initMethodId = 1; %1-TCA, 2-KMM, 3-DG, 4-NNFilter, 5-LR only
percent = 1; % End condition of the pseudo-label refinment procedure. Here, 1 means 100%.
max_iter = 20; % Max iteration of JDM process

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
    
    
    % Traverse target projects
    for i = 1:length(fileList)
        targetName = fileList{i};
        attributeNum = size(eval(targetName),2) - 1;
        labelIndex = size(eval(targetName),2);
        
        targetData=eval(targetName);
        targetData(targetData(:,labelIndex)==-1,labelIndex)=0;
        targetX = targetData(:,1:attributeNum);
        targetX = zscore(targetX);
        targetY = targetData(:,labelIndex);
        
        % Traverse source projects
        for j = 1:length(fileList)
            sourceName = fileList{j};
            attributeNum = size(eval(sourceName),2) - 1;
            labelIndex = size(eval(sourceName),2);
            if(i~=j) % Skip this loop if target and souce projects are same
                sourceData=eval(sourceName);
                sourceData(sourceData(:,labelIndex)==-1,labelIndex)=0;
                sourceX = sourceData(:,1:attributeNum);
                sourceX = zscore(sourceX);
                sourceY = sourceData(:,labelIndex);
                
                % call initMethod to generate init Cls
                [Cls, initMethod] = generateInitCls(initMethodId, sourceX, sourceY, targetX, targetY);
                [~,~,~,~,~,f_measure,~,~,AUC] = evaluate(Cls, targetY);
                
                % save the Cls into ClsArray to compare later
                ClsArray = [Cls];
                for index=1:4
                    learnerName = learnerNames{index};
                    [f_measure,AUC,precision,recall,predictedY] = classifier_example(sourceX,sourceY,targetX,targetY,index);
                    for t = 2:max_iter
                        [betaW, sourceX, sourceY] = JDM('rbf',sourceX,targetX,sourceY,Cls,sigma);
                        betaW = normalizeAlpha(betaW, 1);
                        
                        % Logistic regression
                        %model = train(betaW, Ys, sparse(Xs), '-s 0 -c 1');
                        %Cls = predict(targetY, sparse(targetX), model);
                        %[~,~,~,~,~,f_measure,~,~,AUC] = evaluate(Cls, targetY);
                        
                        [f_measure,AUC,precision,recall,predictedY] = classifier_example(sourceX,sourceY,targetX,targetY,index);
                        
                        % Calculate the percentage of number that same prediction as the previous round
                        size_same = size(Cls(Cls==ClsArray(:,t-1)),1);
                        size_y = size(targetY,1);
                        currentPercent = size_same/size_y;
                        
                        % By comparing last iteration, if all pseudo-labels are not changes, break the loop
                        if currentPercent >= percent
                            break;
                        end
                        
                        ClsArray = [ClsArray,Cls];
                    end
                    %parameter string
                    resultStr = [modelName,',',learnerName,',',dataName,',',targetName,',',sourceName,',',num2str(recall),',',num2str(precision),',',num2str(f_measure),',',num2str(AUC)];
                    fprintf(file,'%s\n',resultStr);
                    disp([modelName,'_',learnerName,'_',dataName,' learning completed！'])
                end
            end
        end
    end
end
end
function [Cls, initMethod] = generateInitCls(initMethodId, sourceX, sourceY, targetX, targetY)

if initMethodId == 1 % use TCA as the initial Pseudo-Label predictor
    initMethod = 'TCA';
    options = tca_options('Kernel', 'linear', 'KernelParam', 1, 'Mu', 1, 'lambda', 1, 'Dim', 10);
    [newtrainX, ~, newtestX] = tca(sourceX, targetX, targetX, options);
    model = train([], sourceY, sparse(newtrainX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(newtestX), model);
elseif initMethodId == 2 % use KMM
    initMethod = 'KMM';
    betaW = KMM('rbf', sourceX, targetX, 0.01);
    betaW = NormalizeAlpha(betaW, 1)
    model = train(betaW, sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 3 % use DG
    initMethod = 'DG';
    gravitation = cal_data_gravitation(targetX, sourceX);
    model = train(gravitation, sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 4 % use NNFilter
    initMethod = 'NNFilter';
    [sourceX, sourceY] = NNFilter(15, sourceX, sourceY, targetX);
    model = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 5 % use LR only
    initMethod = 'LR';
    model = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
end

end