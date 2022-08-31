

### smooth l1 function
smooth_l1<-function(x){
  
  return(sum(x))
}


train<-function(model, 
                x_train,
                y_train,
                kargs){

  #optimizer=optim_sgd(lr=kargs$learning_rate)
  #lr_step(optimizer, step_size = 100, gamma = 0.1)
for (t in 1:kargs$iteration) {
  
  ### -------- Forward pass -------- 
  
  y_pred <- model(x_train)
 
  ### -------- compute loss -------- 
  
  ####### l2 loss of rest of params
  l2_loss=0
  for (i in seq(1,length(model$parameters),2)){
    l2_loss=l2_loss+kargs$lamda_2 * sum(abs(model$linear0)**2)
  }
 
  ####### l1 loss of first layer
  ####### can manipulate 

  l1_loss=kargs$lamda_1 * nnf_smooth_l1_loss(model$linear0,rep(0,model$linear0 %>% length))
  
  
  ####### bce loss
  
  #y_pred=y_pred%>% torch_tensor(dtype = torch_float())
  
  bce_loss=nnf_binary_cross_entropy(y_pred,y_train)
  
  
  ####### total loss
  loss <- bce_loss+l1_loss+l2_loss
  
 
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Zero the gradients before running the backward pass.
  model$zero_grad()
  
  # compute gradient of the loss w.r.t. all learnable parameters of the model
  loss$backward()

  ### -------- Update weights -------- 
  
  # Wrap in with_no_grad() because this is a part we DON'T want to record
  # for automatic gradient computation
  # Update each parameter by its `grad`
  
  with_no_grad({
    model$parameters %>% purrr::walk(function(param) param$sub_(kargs$learning_rate * param$grad))
  })
  
}
  return(model)
}