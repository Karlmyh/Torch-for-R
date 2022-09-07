

creat_model <- nn_module(
  initialize = function(layer_width) {
    torch_manual_seed(1)
    ## suppose there is four layer
    self$linear0 <- torch_randn(layer_width[1],requires_grad = TRUE)
    self$linear1 <- nn_linear(in_features =layer_width[1] , out_features = layer_width[2])
    self$linear2 <- nn_linear(in_features = layer_width[2], out_features = layer_width[3])
    self$linear3<- nn_linear(in_features = layer_width[3], out_features = layer_width[4])
    self$linear4 <- nn_linear(in_features = layer_width[4], out_features = 1)
    
  },
  

  
  forward = function(x) {

      x<-x*self$linear0

      x%>%
      self$linear1() %>%
      nnf_relu()%>%


      self$linear2() %>%
        nnf_relu()%>%

      self$linear3() %>%
        nnf_relu()%>%

      self$linear4() %>%
      nnf_sigmoid()
      # 
      # forward = function(x) {
      #   
      #   m<-nn_tanh()
      #   x<-x*self$linear0
      #   
      #   x%>%
      #     self$linear1() %>%
      #     m%>%
      #     
      #     
      #     self$linear2() %>%
      #     m%>%
      #     
      #     self$linear3() %>%
      #     m%>%
      #     
      #     self$linear4() %>%
      #     nnf_sigmoid()
      
  }
)
