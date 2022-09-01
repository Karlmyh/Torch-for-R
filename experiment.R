absolute_path<-"/Users/mayuheng/Documents/GitHub/Torch-for-R"
setwd(absolute_path)

library(torch)
torch_set_default_dtype(torch_float())
source("data.R")
source("model.R")
source("train.R")
source("evaluation.R")

experiment_set=1

# prepare data
prepared_data<-generate_data()
x_train=prepared_data[[1]][[experiment_set]] %>% torch_tensor(dtype = torch_float())
y_train=prepared_data[[2]][[experiment_set]] %>% torch_tensor(dtype = torch_float())
x_test=prepared_data[[3]][[experiment_set]]  %>% torch_tensor(dtype = torch_float())
y_test=prepared_data[[4]][[experiment_set]]  


# creat model
layer_width<-c(100,200,10)
model<-creat_model(layer_width)

# hyperparameters
kargs<-NULL
kargs$learning_rate <- 0.1
kargs$lamda_1=0.5
kargs$lamda_2=0.01
kargs$iteration=400

# train model
trained_model<-train(model, x_train,y_train,kargs)

# evaluate model 
accuracy<-evaluation(trained_model,x_test,y_test)

accuracy
