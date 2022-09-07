library(caret)
evaluation<-function(trained_model,
                     x_test_1,
                     x_test_2,
                     x_test_3,
                     y_test_1,
                     y_test_2,
                     y_test_3){
  y_prob_1=as_array(trained_model[[1]](x_test_1))
  y_prob_2=as_array(trained_model[[2]](x_test_2))
  y_prob_3=as_array(trained_model[[3]](x_test_3))
  
  
  y_pred_1=(y_prob_1>0.5) %>% as.integer
  y_pred_2=(y_prob_2>0.5) %>% as.integer
  y_pred_3=(y_prob_3>0.5) %>% as.integer
  
  pred=as.factor(c(y_pred_1,y_pred_2,y_pred_3))
  actual=as.factor(c(as_array(y_test_1),as_array(y_test_2),as_array(y_test_3)))
  levels(pred)<-c("0","1")
  
  
  accuracy<-(pred==actual) %>% mean
  result_matrix<-confusionMatrix(pred,actual)
  
  return(result_matrix)
}


Eva_variable_selection<- function(index, beta){
  ##### Input: #####
  # index: important variables selected by the network for the i-th dataset
  #	beta: model coefficients of the i-th dataset
  ##### Output: #####
  # Evaluation Index (SEN, SPE, GM, CCR) for the i-th dataset.
  
  trueindex=which(beta!=0)
  TP=0
  FP=0
  FN=0
  TN=0
  TP=length(intersect(index,trueindex))
  FP=length(setdiff(index,trueindex))
  FN=length(trueindex)-TP
  TN=p-length(trueindex)-FP
  SEN=TP/(TP+FN)
  SPE=TN/(TN+FP)
  GM=sqrt(SEN*SPE)
  MR=(FP+FN)/(TP+FN+TN+FP)
  CCR=1-MR
  eva_index=return(c(SEN,SPE,GM,CCR))
}