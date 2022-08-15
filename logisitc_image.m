%%LOGISTIC REGRESSION ON THE IMAGE DATA

%%
%EXTRACT THE DATA THAT CAN BE USED 
clear all;
clc;
close all;

%extracting train and dev image_data
data_path = 'data\image_data'; 
sp_train=cell(1,5);
sp_develop=cell(1,5);

%finding the directory of the given datasets
%1=coast,%2=highway,%3=insidecity,%4=opencountry,%5=tallbuilding
count=1;
count_train=zeros(1,5);
count_develop=zeros(1,5);


for i=3:7
    dlist=dir('data\image_data'); 
    c=dlist(i).name;
    
    t_file=fullfile(data_path,c,'train');
    d_file=fullfile(data_path,c,'dev');
    
    tdata_list= dir(fullfile(t_file, '*.jpg_color_edh_entropy'));
    ddata_list=dir(fullfile(d_file, '*.jpg_color_edh_entropy'));
   
    tdata=cell(1,length(tdata_list));
    ddata=cell(1,length(ddata_list));
     
    im_train{count}=tdata;
     im_develop{count}=ddata;
    
    for j=1:length(tdata_list)
        tmfc=fullfile(t_file,tdata_list(j).name);
        format long g
        im_train{count}{j} =dlmread(tmfc);
        count_train(1,i-2)=count_train(1,i-2)+1;
    end
    
    for j=1:length(ddata_list)
        tmfc=fullfile(d_file,ddata_list(j).name);
        format long g
        im_develop{count}{j} =dlmread(tmfc);
        count_develop(1,i-2)=count_develop(1,i-2)+1;
    end  
count=count+1;

end
%%
A=reshape(im_train{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_train{i})
        a=reshape(im_train{i}{j}',[],1);
        A=[A a];
    end
end
A=A(:,2:end)';
%%MAKE USE OF THE FLDA FOR THE PURPOSE OF REDUCING THE DIMENSION
%%
%A is the matrix having all the 36X23 matrix as the column vectors
[coeff,score,~,~,explained]=pca(A);
%%
%within class covariance normalisation of the score data
count1=1;
for i=1:1
    A_with=zeros(count_train(i),828);
    for j=1:count_train(i)
        A_with(j,:)=A(count1,:);
        count1=count1+1;
    end
    score_norm=normalize(A_with,2);
end
for i=2:5
    A_with=zeros(count_train(i),828);
    for j=1:count_train(i)
        A_with(j,:)=A(count1,:);
        count1=count1+1;
    end
    score_norm=[score_norm ;normalize(A_with,2)];
end
%%
% %finding the mean of each class separately
% 
mn_vec=zeros(5,828);
id=1;
for i=1:5
    sum=0;
    for j=1:length(im_train{i})
        sum=sum+score_norm(id,:);
        id=id+1;
    end
    mn_vec(i,:)=sum/length(im_train{i});

end
%%
%within class matrix for the classes
S_w=zeros(828,828);
idx=1;
%finding the total scatter matrix
[v,~]=size(score_norm);
m=zeros(v,1);
for i=1:v
    m=m+score_norm(i,:);
end
m_n=m/v;
S_t=zeros(828,828);
for j=1:v
    S_t=S_t+(score_norm(j,:)'-m_n')*(score_norm(j,:)'-m_n')';
end
%%
 idx=1;  
for i=1:5
    S1=zeros(828,828);
    for j=1:length(im_train{i})
        S1=score_norm(idx,:)'-mn_vec(i,:)';
        idx=idx+1;
    end
    S_w=S_w+S1*S1';
end
S_b=S_t-S_w;
epsilon=0.0001;
S_w=S_w+epsilon;
[V D]=eig(inv(S_w)*S_b);
m=4;
V_s=zeros(828,m);
for i=1:m
  V_s(:,i)=V(:,i);
end
B=A*V_s;
%%
%making the logistic regression classifier
[~,n]=size(score_norm);
weight=zeros(5,n+1);
k=1;
idx=1;
T=zeros(1184,1);
for i=1:5
    t=zeros(1184,1);
    for j=1:count_train(i)
     if i==k
         t(idx,1)=1;
     end
     T(idx,1)=i;
     idx=idx+1;
    end
    k=k+1;
    weight(i,:)=logistic(score_norm,t);
end

%weight(1,:)=logistic(A(1:433,:),T-1);
%% 
%for tuning of the parameter
dev_u=reshape(im_develop{1}{1}',[],1);
     
for i=1:5
    for j=1:length(im_develop{i})
        a=reshape(im_develop{i}{j}',[],1);
        dev_u=[dev_u a];
    end
end
dev_u=dev_u(:,2:end)';
%%dev_uu=dev_u*V_s;
dev_ud=[ones(340,1) dev_u];
%%
dev_prob=zeros(340,5);
predict=zeros(340,5);
predict_id=zeros(340,1);
idx=1;
for i=1:5
    for j=1:length(im_develop{i})
        
        dev_prob(idx,:)=(dev_ud(idx,:)*weight');
        
       predict(idx,:)=sigmoid_multc(dev_prob(idx,:));
       predict_id(idx)=find(predict(idx,:)==max(predict(idx,:)));
        idx=idx+1;
    end
end
 %%
 actual=zeros(340,1);
 idx=1;
 for i=1:5
     for j=1:length(im_develop{i})
         actual(idx)=i;
         idx=idx+1;
     end
 end
 %%

%%MAKE THE CONFUSION MATRIX FOR THE GIVEN CLASSIFIER
confusion_matrix(actual',predict_id');
%%
%plot the roc curve for the flda image
target=zeros(340,5);
co=1;
for i=1:5
    for j=1:count_develop(i)
        target(co,i)=1;
        co=co+1;
    end
end
[tpr fpr thersholds]=roc(target,predict);
plotroc(tpr,fpr);
%%
%plot the det plot for the flda imaeg plot
targ=zeros(340,1);
nontrag=zeros(4*340,1);
k=1;
co=1;
ll=1;
for i=1:5
    for j=1:count_develop(i)
        for k=1:5
            if i==k
              targ(co,1)=predict(co,i);
            end
            if i~=k
              nontarg(ll,1)=predict(co,k);
              ll=ll+1;
            end
        end
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
          