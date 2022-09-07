
library(MASS)

DataDiv <- function(data, trainrate,n1,n2,n3){
  # Divide each dataset into training set and test set. The training set is used to train a PIN network, and the test set is used to select important variables for new data.
  x1=data$x1
  x2=data$x2
  x3=data$x3
  
  y1=data$y1
  y2=data$y2
  y3=data$y3
  
  nn1=trainrate*n1
  nn2=trainrate*n2
  nn3=trainrate*n3
  
  train_id_1=sample(1:n1,nn1)
  test_id_1=setdiff(1:n1,train_id_1)
  
  train_id_2=sample(1:n2,nn2)
  test_id_2=setdiff(1:n2,train_id_2)
  
  train_id_3=sample(1:n3,nn3)
  test_id_3=setdiff(1:n3,train_id_3)
  
  x_train_yu=list(x1[train_id_1,],x2[train_id_2,],x3[train_id_3,])
  x_test_yu=list(x1[test_id_1,],x2[test_id_2,],x3[test_id_3,])
  
  y_train_yu=list(y1[train_id_1],y2[train_id_2],y3[train_id_3])
  y_test_yu=list(y1[test_id_1],y2[test_id_2],y3[test_id_3])
  
  x=list(x1,x2,x3)
  y=list(y1,y2,y3)
  
  dataDiv <- list(x_train_yu, x_test_yu, y_train_yu, y_test_yu, x, y)
  return(dataDiv)		
}

DataC <- function(dataDiv){
  # Preprocess the X and Y in the training set and test set of each data set separately.
  
  x_train_yu=dataDiv[[1]]
  x_test_yu=dataDiv[[2]]
  y_train_yu=dataDiv[[3]]
  y_test_yu=dataDiv[[4]]
  x=dataDiv[[5]]
  y=dataDiv[[6]]
  
  # For X
  x_train=list()
  for(i in 1:length(x_train_yu)){
    x_train[[i]]<-scale(x_train_yu[[i]],center = T,scale=T)
  }
  x_test=list()
  for(i in 1:length(x_test_yu)){
    x_test[[i]]<-scale(x_test_yu[[i]],center = T,scale=T)
  }
  
  # For Y
  y_train=y_train_yu
  y_test=y_test_yu
  
  # Return
  dataC <- list(x_train, y_train, x_test, y_test)
  return(dataC)
}
DataHoLinear <- function(n1, n2, n3, b1, b2, b3, p, corval){
  # n1: sample size of dataset 1
  # n2: sample size of dataset 2
  # n3: sample size of dataset 3
  # b1: model coefficients of dataset 1
  # b2: model coefficients of dataset 1
  # b3: model coefficients of dataset 1
  # p : dimensions
  # corval: correlation coefficient of power structure
  # 
  # Output: 
  # This function will result three datasets with size n1, n2,and n3 respectively.
  mean=matrix(c(rep(0,p)),nrow=p)
  sigma=diag(p)
  for(sid in 1:p){
    for(sidd in 1:p){
      sigma[sid,sidd]=corval^abs(sid-sidd) # Power structure
    }
  }
  
  ## X generation
  set.seed(1)
  x1<-mvrnorm(n1,mean,sigma)
  x2<-mvrnorm(n2,mean,sigma)
  x3<-mvrnorm(n3,mean,sigma)
  
  # SNR=3
  
  s1=x1%*%b1
  s2=x2%*%b2
  s3=x3%*%b3
  
  ## Y generation
  
  sig_threshold=0.5
  sigmod = function(inX)
  {
    return (1 / (1 + exp(-inX)))
  }
  
  Pi_test_1= sigmod(x1%*%b1)
  Pi_test_2= sigmod(x2%*%b2)
  Pi_test_3= sigmod(x3%*%b3)
  
  ## set seed
  y1= rbinom(length(Pi_test_1), 1, Pi_test_1)
  y2= rbinom(length(Pi_test_2), 1, Pi_test_2)
  y3= rbinom(length(Pi_test_3), 1, Pi_test_3)
  
  data <- list(x1=x1, x2=x2, x3=x3, y1=y1, y2=y2, y3=y3)
  return(data)		
}


generate_data<-function(){
n1=200
n2=175
n3=150

p=100 # Each study has the same number of variables
corval=0 # Covariance of the covariate are AR(0) structure

b1=matrix(c(4,4,-8,-4,2,8,-4,4,-2,4,rep(0,90)),ncol = 1)
b2=matrix(c(-2,-2,4,-2,1,-4,2,-2,-1,2,rep(0,90)),ncol = 1)
b3=matrix(c(-1.5,-1.5,3,-1.5,-0.75,-3,1.5,-1.5,0.75,1.5,rep(0,90)),ncol = 1)

data=DataHoLinear(n1, n2, n3, b1, b2, b3, p, corval)
trainrate=0.8
dataDiv=DataDiv(data, trainrate,n1,n2,n3)
dataClean=DataC(dataDiv)


return(dataClean)
}
