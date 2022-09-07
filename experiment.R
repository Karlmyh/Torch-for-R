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



# train model

trained_model_list<-train(model_1,model_2,model_3,x_train_1,x_train_2,x_train_3,y_train_1,y_train_2,y_train_3,kargs)$model_list

y_prior_1=as_array(trained_model_list[[1]](x_train_1))
y_prior_2=as_array(trained_model_list[[2]](x_train_2))
y_prior_3=as_array(trained_model_list[[3]](x_train_3))

rm(model_1)
rm(model_2)
rm(model_3)
rm(trained_model_list)


kargs$eta= seq(from = 0.05, to = 0.5, by = 0.1)

output_bic = vector()
for (j in 1:length(kargs$eta)) {
  y_post_1=((1-kargs$eta[j])*as_array(y_train_1) +kargs$eta[j]*y_prior_1)%>% torch_tensor(dtype = torch_float())
  y_post_2=((1-kargs$eta[j])*as_array(y_train_2)+kargs$eta[j]*y_prior_2)%>% torch_tensor(dtype = torch_float())
  y_post_3=((1-kargs$eta[j])*as_array(y_train_3)+kargs$eta[j]*y_prior_3)%>% torch_tensor(dtype = torch_float())

#as_array(y_train_1==y_post_1)

  model_post_1<-creat_model(layer_width)
  model_post_2<-creat_model(layer_width)
  model_post_3<-creat_model(layer_width)
  
  ans_post_model_list<-train(model_post_1,model_post_2,model_post_3,x_train_1,x_train_2,x_train_3,y_post_1,y_post_2,y_post_3,kargs,post_train=1)
  loss=ans_post_model_list$loss
  num_w=length(which(ans_post_model_list$w_change_list[[1]]!=0))
  rm(model_post_1)
  rm(model_post_2)
  rm(model_post_3)
  rm(ans_post_model_list)
  
  
  BIC=2*log(loss)+(log(n1+n2+n3))*(num_w+p*q+q*q*(m-1)+q*1+m*q+1)
  output_bic=c(output_bic,BIC)
}


kargs$final_eta=kargs$eta[which.min(output_bic)]

model_final_post_1<-creat_model(layer_width)
model_final_post_2<-creat_model(layer_width)
model_final_post_3<-creat_model(layer_width)

y_final_post_1=((1-kargs$final_eta)*as_array(y_train_1) +kargs$final_eta*y_prior_1)%>% torch_tensor(dtype = torch_float())
y_final_post_2=((1-kargs$final_eta)*as_array(y_train_2)+kargs$final_eta*y_prior_2)%>% torch_tensor(dtype = torch_float())
y_final_post_3=((1-kargs$final_eta)*as_array(y_train_3)+kargs$final_eta*y_prior_3)%>% torch_tensor(dtype = torch_float())

final_post_model_list<-train(model_final_post_1,model_final_post_2,model_final_post_3,x_train_1,x_train_2,x_train_3,y_final_post_1,y_final_post_2,y_final_post_3,kargs,post_train=1)$model_list


# evaluate model 
cmtx<-evaluation(final_post_model_list,x_test_1,x_test_2,x_test_3,y_test_1,y_test_2,y_test_3)
index_5=which(final_post_model_list$w_change_list[[1]]!=0)
index_5

rm(model_final_post_1)
rm(model_final_post_2)
rm(model_final_post_3)
rm(final_post_model_list)

print(cmtx)
TPR=cmtx$table[1,1]/apply(cmtx$table,2,sum)[1]
TNR=cmtx$table[2,2]/apply(cmtx$table,2,sum)[2]
print(sqrt(TPR*TNR))



##### comparison method
# 
# model_without_prior_1<-creat_model(layer_width)
# model_without_prior_2<-creat_model(layer_width)
# model_without_prior_3<-creat_model(layer_width)
# 
# ans_without_prior<-train(model_without_prior_1,model_without_prior_2,model_without_prior_3,x_train_1,x_train_2,x_train_3,y_train_1,y_train_2,y_train_3,kargs,post_train=1)
# trained_model_without_prior_list<-ans_without_prior$model_list
# 
# # evaluate model 
# cmtx_no_prior<-evaluation(trained_model_without_prior_list,x_test_1,x_test_2,x_test_3,y_test_1,y_test_2,y_test_3)

print(cmtx_no_prior)
TPR_no_prior=cmtx_no_prior$table[1,1]/apply(cmtx_no_prior$table,2,sum)[1]
TNR_no_prior=cmtx_no_prior$table[2,2]/apply(cmtx_no_prior$table,2,sum)[2]

print(sqrt(TPR*TNR))
print(sqrt(TPR_no_prior*TNR_no_prior))

remove(model_without_prior_1)
remove(model_without_prior_2)
remove(model_without_prior_3)
remove(ans_without_prior)
remove(trained_model_without_prior_list)
