%%LOGISTICS REGRESSION ON THE SYNTHETIC DATA
%%

%EXTRACTING THE DATASETS FOR THE SYNTHETIC DATA
clear all;
clc;
close all;

data_path = 'data/Synthetic_Data/';

tdata_ind = 1;
ddata_ind = 1;
tname = strcat(data_path, '/train.txt');
dname = strcat(data_path, '/dev.txt');

fid = fopen(tname,'r');
xx = textscan(fid,'%f,%f,%d');
fclose(fid);
tdata = [xx{1} xx{2}];

tnum = length(xx{1});
%extracting the develop data
fid = fopen(dname,'r');
yy = textscan(fid,'%f,%f,%d');
fclose(fid);
ddata = [yy{1} yy{2}];

dnum = length(yy{1});

%%

count_train=zeros(1,2);

for i=1:length(xx{3})
    if xx{3}(i)==1
        count_train(1,1)=count_train(1,1)+1;
    end
    if xx{3}(i)==2
        count_train(1,2)=count_train(1,2)+1;
    end
end
count1=1;
for i=1:1
    A_with=zeros(count_train(i),2);
    for j=1:count_train(i)
        A_with(j,:)=tdata(count1,:);
        count1=count1+1;
    end
    score_norm=A_with./max(A_with);
end
for i=2:2
    A_with=zeros(count_train(i),2);
    for j=1:count_train(i)
        A_with(j,:)=tdata(count1,:);
        count1=count1+1;
    end
    score_norm=[score_norm ;A_with./max(A_with)];
    
end
%%
%making the logistic regression classifier
T=zeros(tnum,2);
for i=1:tnum
    if xx{3}(i)==1
        T(i,1)=1;
    end
    if xx{3}(i)==2
        T(i,2)=1;
    end
end
for i=1:2
    weight(i,:)=logistic(score_norm,T(:,i));
end

%score=[ones(length(score_norm),1) score_norm];
%%
%%APPLY THE LOGISITIC REGRESSION CLASSIFIER ON THE DATASETS
d_data=[ones(dnum,1) ddata];

predict_data=zeros(dnum,2);
predict_id=zeros(dnum,1);
idx=1;
for i=1:2
    predict_data(:,i)=sigmoid(d_data*weight(i,:)');
end
for i=1:dnum
     predict_id(i)=find(predict_data(i,:)==max(predict_data(i,:)));
end
%%
%Make the confusion matrix for the logistic regression on the synthetic data
confusion_matrix(double(yy{3})',predict_id');
%%
%plot the roc curve for the flda synthetic data
targets=zeros(dnum,2);
j=1;
for i=1:dnum
    if yy{3}(i)==1
        targets(i,1)=1;
    end
    if yy{3}(i)==2
        targets(i,2)=1;
    end
end
[tpr fpr thersholds]=roc(targets,predict_data);
plotroc(tpr,fpr);

%%
%plot  det curve for the flda syntheticc class
targ=zeros(1000,1);
nontarg=zeros(1000,1);
for i=1:dnum
    if yy{3}(i)==1
        targ(i,1)=predict_data(i,1);
        nontarg(i,1)=predict_data(i,2);
    end
    if yy{3}(i)==2
        targ(i,1)=predict_data(i,2);
        nontarg(i,1)=predict_data(i,1);
    end
end

plot_title = 'DET plot example';
prior = 0.3;

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);

plot_obj.set_system(targ',nontarg','hw');
plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
plot_obj.plot_DR30_fa('c--','30 false alarms');
plot_obj.plot_DR30_miss('k--','30 misses');
plot_obj.plot_mindcf_point(prior,{'b*','MarkerSize',8},'mindcf');

 plot_obj.set_system(targ',nontarg','hw123');
plot_obj.plot_steppy_det({'r','LineWidth',2},' ');
plot_obj.plot_DR30_fa('m--','30 false alarms');
plot_obj.plot_DR30_miss('g--','30 misses');
 plot_obj.plot_mindcf_point(prior,{'r*','MarkerSize',8},'mindcf');

plot_obj.display_legend();
   
