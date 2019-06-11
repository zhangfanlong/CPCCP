function [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC, AUC] = evaluate(PREDICTED, ACTUAL)
    [label_num_1,accuracy_1,sensitivity_1,specificity_1,precision_1,recall_1,f_measure_1,gmean_1,MCC_1, AUC_1] = evaluate_each(PREDICTED, ACTUAL, 0);
    [label_num_2,accuracy_2,sensitivity_2,specificity_2,precision_2,recall_2,f_measure_2,gmean_2,MCC_2, AUC_2] = evaluate_each(PREDICTED, ACTUAL, 1);
    accuracy = average(accuracy_1, label_num_1,accuracy_2, label_num_2);
    sensitivity = average(sensitivity_1, label_num_1,sensitivity_2, label_num_2);
    specificity = average(specificity_1, label_num_1,specificity_2, label_num_2);
    precision = average(precision_1, label_num_1,precision_2, label_num_2);
    recall = average(recall_1, label_num_1,recall_2, label_num_2);
    f_measure =average(f_measure_1, label_num_1,f_measure_2, label_num_2);
    gmean = average(gmean_1, label_num_1,gmean_2, label_num_2);
    MCC = average(MCC_1, label_num_1,MCC_2, label_num_2);
    AUC =average(AUC_1, label_num_1,AUC_2, label_num_2);    
end

function [average_value] = average(value_1,num_1, value_2, num_2)
    average_value = (value_1*num_1 +  value_2*num_2)/(num_1+ num_2);
end

function [label_num,accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC, AUC] = evaluate_each(PREDICTED, ACTUAL, LABEL)

idx = (ACTUAL()==LABEL);

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;

label_num = p;

tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp; 

tp_rate = tp/p;
tn_rate = tn/p;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
MCC = (tp*tn - fp*fn) /  sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

%º∆À„AUC÷µ
[A,I]=sort(PREDICTED);
M=0;
N=0;
for i=1:length(PREDICTED)
    if(ACTUAL(i)==1)
        M=M+1;
    else
        N=N+1;
    end
end
sigma=0;
for i=M+N:-1:1
    if(ACTUAL(I(i))==1)
        sigma=sigma+i;
    end
end
AUC=(sigma-(M+1)*M/2)/(M*N);

if isnan(f_measure)
    f_measure = 0;
end
    
if isnan(MCC)
    MCC = 0;
end

if isnan(AUC)
    AUC = 0;
end
    
end