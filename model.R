
creat_model <- nn_module(
  initialize = function(layer_width) {
    
    ## suppose there is three layer
    self$linear0 <- torch_randn(layer_width[1],requires_grad = TRUE)
    self$linear1 <- nn_linear(in_features =layer_width[1] , out_features = layer_width[2])
    self$linear2 <- nn_linear(in_features = layer_width[2], out_features = layer_width[3])
    self$linear3 <- nn_linear(in_features = layer_width[3], out_features = 1)
    
  },
  
  forward = function(x) {
      x<-x*self$linear0
      
      x%>%
      self$linear1() %>%
      nnf_relu() %>%
      
      
      self$linear2() %>%
      nnf_relu() %>%
      
      self$linear3() %>%
      nnf_sigmoid()
      
      
  }
)