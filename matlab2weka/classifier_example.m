function [f_measure,AUC,precision,recall,predictedY] = classifier_example(newtrainX,sourceY,newtestX,targetY,choice)
% A example code that runs a classification algorithm on the IRIS dataset.
% http://www.sunghoonivanlee.com
% Written by Sunghoon Ivan Lee, All copy rights reserved, 2/20/2015 
% Revised on 7/22/2015



%% Initializing 
% adding the path to matlab2weka codes

% adding Weka Jar file

% adding matlab2weka JAR file that converts the matlab matrices (and cells)
% to Weka instances.



% numerical class variable

%feat_num = iris(:,1:4);
featName = {'1', '2', '3', '4','5','6','7','8','9','10'};

% converting to nominal variables (Weka cannot classify numerical classes)
class_source = cell(size(sourceY));
class_target = cell(size(targetY));
uClass_source = unique(sourceY);
uClass_target = unique(targetY)
tmp_source = cell(1,1);
for i = 1:length(uClass_source)
    tmp_source{1,1} = strcat('class_', num2str(i-1));
    class_source(sourceY == uClass_source(i),:) = repmat(tmp_source, sum(sourceY == uClass_source(i)), 1);
end
temp_target = cell(1,1);
for i = 1:length(uClass_target)
    temp_target{1,1} = strcat('class_', num2str(i-1));
    class_target(targetY == uClass_target(i),:) = repmat(temp_target, sum(targetY == uClass_target(i)), 1);
end
clear uClass_source uClass_target tmp_source temp_target i

% Choosing a regression tool to be used
% -------------------------------------
% classifier = 1: Random Forest Classifier from WEKA
% classifier = 2: Gaussian Process Regression from WEKA
% classifier = 3: Support Vector Machine from WEKA
% classifier = 4: Logistic Regression from WEKA
classifier = choice;

%% Performing K-fold Cross Validation
%K = 10;
%N = size(feat_num,1);
%indices for cross validation
%idxCV = ceil(rand([1 N])*K); 
actualClass = cell(size(newtrainX,1),1);
predictedClass = cell(size(newtestX,1),1);
%for k = 1:K
    %defining training and testing sets
feature_train = newtrainX;
class_train = class_source;
feature_test = newtestX;
class_test = class_target;

%performing regression
[actual_tmp, predicted_tmp, probDistr_tmp] = wekaClassification(feature_train, class_train, feature_test, class_test, featName, classifier);

%accumulating the results
actualClass = actual_tmp;
predictedClass = predicted_tmp;    


predictedY = zeros(size(predictedClass));
for i = 1:length(predictedClass)
    if(predictedClass{i} == 'class_1')
        predictedY(i,1) = 1;
    else
        predictedY(i,1) = 0;
    end
end

actualY = zeros(size(actualClass));
for i = 1:length(actualClass)
    if(actualClass{i} == 'class_1')
        actualY(i,1) = 1;
    else
        actualY(i,1) = 0;
    end
end


[accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC, AUC] = evaluate(predictedY, actualY)

end
%clear idxCV k
