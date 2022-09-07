absolute_path<-"C:/Users/jiake/Desktop/融合先验信息的pin/Torch-for-R"
setwd(absolute_path)
rm(list = ls())
library(torch)
torch_set_default_dtype(torch_float())
source("data.R")
source("model.R")
source("train.R")
source("evaluation.R")


# prepare data
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

# hyperparameters
kargs<-NULL
kargs$learning_rate <-0.5
kargs$decay_iter=200
kargs$decay_alpha=0.2
kargs$lamda_1=0.06#tuning parameter for variable selection in the objective function
kargs$lamda_2=0.0005# tuning parameter for preventing overfitting in the objective function
kargs$iteration=600
kargs$p=5 #prior=c(1:5)


# creat model
p=100
q=10
m=3
n1=200
n2=175
n3=150
layer_width<-c(100,10,10,10)
model_1<-creat_model(layer_width)
model_2<-creat_model(layer_width)
model_3<-creat_model(layer_width)


##### comparison method

model_without_prior_1<-creat_model(layer_width)
model_without_prior_2<-creat_model(layer_width)
model_without_prior_3<-creat_model(layer_width)

ans_without_prior<-train(model_without_prior_1,model_without_prior_2,model_without_prior_3,x_train_1,x_train_2,x_train_3,y_train_1,y_train_2,y_train_3,kargs,post_train=1)
trained_model_without_prior_list<-ans_without_prior$model_list
index_0=which(ans_without_prior$w_change_list[[1]]!=0)
# evaluate model 
cmtx_no_prior<-evaluation(trained_model_without_prior_list,x_test_1,x_test_2,x_test_3,y_test_1,y_test_2,y_test_3)

print(cmtx_no_prior)
TPR_no_prior=cmtx_no_prior$table[1,1]/apply(cmtx_no_prior$table,2,sum)[1]
TNR_no_prior=cmtx_no_prior$table[2,2]/apply(cmtx_no_prior$table,2,sum)[2]

# print(sqrt(TPR*TNR))
print(sqrt(TPR_no_prior*TNR_no_prior))
index_0

remove(model_without_prior_1)
remove(model_without_prior_2)
remove(model_without_prior_3)
remove(ans_without_prior)
remove(trained_model_without_prior_list)
