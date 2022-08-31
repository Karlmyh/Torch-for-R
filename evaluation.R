
evaluation<-function(trained_model,
                     x_test,
                     y_test){
  y_prob=as_array(trained_model(x_test))
  y_pred=(y_prob>0.5) %>% as.integer
  accuracy<-(y_pred==y_test) %>% mean
  return(accuracy)
}