rm(list = ls())
library(torch)
library(parallel)
library(caret)
library(MASS)
torch_set_default_dtype(torch_float())
source("data.R")
source("model.R")
source("train.R")
source("evaluation.R")


# prepare data
b1=matrix(c(4,4,-8,-4,2,8,-4,4,-2,4,rep(0,90)),ncol = 1)
b2=matrix(c(-2,-2,4,-2,1,-4,2,-2,-1,2,rep(0,90)),ncol = 1)
b3=matrix(c(-1.5,-1.5,3,-1.5,-0.75,-3,1.5,-1.5,0.75,1.5,rep(0,90)),ncol = 1)

# creat model
p=100
q=10
m=3
n1=200
n2=175
n3=150
layer_width<-c(100,10,10,10)
n.simu=4
variable_selection_mat = matrix(NA, 1, p)


my.Simu <- function(s){
  
  prepared_data<-generate_data()
  x_train_1=prepared_data[[1]][[1]]%>% torch_tensor(dtype = torch_float())
  y_train_1=prepared_data[[2]][[1]]%>% torch_tensor(dtype = torch_float())
  x_test_1=prepared_data[[3]][[1]]%>% torch_tensor(dtype = torch_float())
  y_test_1=prepared_data[[4]][[1]]%>% torch_tensor(dtype = torch_float())
  
  x_train_2=prepared_data[[1]][[2]]%>% torch_tensor(dtype = torch_float())
  y_train_2=prepared_data[[2]][[2]]%>% torch_tensor(dtype = torch_float())
  x_test_2=prepared_data[[3]][[2]]%>% torch_tensor(dtype = torch_float())
  y_test_2=prepared_data[[4]][[2]]%>% torch_tensor(dtype = torch_float())
  
  x_train_3=prepared_data[[1]][[3]]%>% torch_tensor(dtype = torch_float())
  y_train_3=prepared_data[[2]][[3]]%>% torch_tensor(dtype = torch_float())
  x_test_3=prepared_data[[3]][[3]]%>% torch_tensor(dtype = torch_float())
  y_test_3=prepared_data[[4]][[3]]%>% torch_tensor(dtype = torch_float())
  
  set.seed(s)
  
  
  # train model
  model_without_prior_1<-creat_model(layer_width)
  model_without_prior_2<-creat_model(layer_width)
  model_without_prior_3<-creat_model(layer_width)
  
  # prior hyperparameters
  kargs<-NULL
  kargs$learning_rate <-0.5
  kargs$decay_iter=200
  kargs$decay_alpha=0.2
  kargs$lamda_1=0.06#tuning parameter for variable selection in the objective function
  kargs$lamda_2=0.0005# tuning parameter for preventing overfitting in the objective function
  kargs$iteration=600
  kargs$p=0 #non prior
  
  ans_without_prior<-train(model_without_prior_1,model_without_prior_2,model_without_prior_3,x_train_1,x_train_2,x_train_3,y_train_1,y_train_2,y_train_3,kargs,post_train=1)
  trained_model_without_prior_list<-ans_without_prior$model_list
  

  
  
  # evaluate model 

  cmtx_no_prior<-evaluation(trained_model_without_prior_list,x_test_1,x_test_2,x_test_3,y_test_1,y_test_2,y_test_3)
  index_0=which(ans_without_prior$w_change_list[[1]]!=0)
  eva=Eva_variable_selection(index_0,b1)
  
  variable_selection_mat[index_0] =1
  variable_selection_mat[setdiff(1:100,index_0)] =0
  
  rm(model_final_post_1)
  rm(model_final_post_2)
  rm(model_final_post_3)
  rm(final_post_model_list)
  
  TPR_no_prior=cmtx_no_prior$table[1,1]/apply(cmtx_no_prior$table,2,sum)[1]
  TNR_no_prior=cmtx_no_prior$table[2,2]/apply(cmtx_no_prior$table,2,sum)[2]
  GM=sqrt(TPR_no_prior*TNR_no_prior)
  pre_eva=c(GM,TPR_no_prior,TNR_no_prior)
  
  final_result=list(pre_eva=pre_eva,eva=eva,index=index_0,variable_selection_mat=variable_selection_mat,pre_gmeans=GM,final_ita=kargs$final_eta)
  
  return(final_result)
}



no_cores <- detectCores()/4
#clusterExport(cl,varlist=c("DataHoLinear","DataDiv","DataC","dtanh","sigmod"),envir = environment())
cl<- makeCluster(no_cores)
clusterEvalQ(cl,{library(MASS)}) 
clusterEvalQ(cl,{library(torch)})
clusterEvalQ(cl,{library(caret)})
clusterExport(cl, ls())

non_prior_results<-parLapply(cl, 1:n.simu, my.Simu)
stopCluster(cl) # 关闭集群

variable_selection_result= matrix(0, 1, p)
pre_eva_result=matrix(0,n.simu,3)
eva_result=matrix(0,n.simu,4)
seperate_gmeans_result=matrix(0,n.simu,m)


for(i in 1:n.simu){
  variable_selection_result=variable_selection_result+t(matrix(non_prior_results[[i]]$variable_selection_mat))
  pre_eva_result[i,]=as.vector(non_prior_results[[i]]$pre_eva)
  eva_result[i,]=as.vector(non_prior_results[[i]]$eva)
  seperate_gmeans_result[i,]=as.vector(non_prior_results[[i]]$pre_gmeans)
}


seperate_gmeans=matrix(apply(seperate_gmeans_result, 2, mean),nrow = 1)

ho_variable_selection_result = matrix(apply(eva_result,2,mean),nrow = 1)
ho_prediction_result=matrix(apply(pre_eva_result,2,mean),nrow = 1)

colnames(ho_variable_selection_result) = c("SEN","SPE","GM","CCR")
colnames(ho_prediction_result) = c("PRE-GM","PRE-SEN","PRE-SPE")
colnames(seperate_gmeans) = c("data_1","data_2","data_3")

ho_variable_selection_result_0=ho_variable_selection_result
ho_prediction_result_0=ho_prediction_result
seperate_gmeans_0=seperate_gmeans
variable_selection_result_0=variable_selection_result

ho_variable_selection_result_0
ho_prediction_result_0
variable_selection_result_0
seperate_gmeans_0

write.table(ho_variable_selection_result_0,file="ho_variable_selection_result_0.txt")
write.table(ho_prediction_result_0,file="ho_prediction_result_0.txt")
write.table(variable_selection_result_0,file="variable_selection_result_0.txt")
write.table(seperate_gmeans_0,file="seperate_gmeans_0.txt")
