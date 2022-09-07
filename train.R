

### smooth l1 function
smooth_absolute<-function(sqsum){
  a=0.04
  var_1=min(sqsum,a)
  var_2=max(sqsum,a)
  loss_1=-var_1**4/(8*a**3)+3*var_1**2/(4*a)+3*var_1/8-a
  loss_2= var_2
  return(loss_1+loss_2)
}

smooth_l1<-function(list_x,p){
  
  
  num_x=length(list_x)
  
  loss=0
  
  for (i in (p+1):length(list_x[[1]])){
    sumsq=0
    for (j in 1:num_x) {
      sumsq=sumsq+list_x[[j]][i]**2
    }
    sqsum=sqrt(sumsq)
    loss_temp=smooth_absolute(sqsum)
    loss=loss+loss_temp
  }
  return(loss)
}


train<-function(model_1,model_2,model_3,x_train_1,x_train_2,x_train_3,y_train_1,y_train_2,y_train_3,kargs,post_train=0){

  #optimizer=optim_sgd(lr=kargs$learning_rate)
  #lr_step(optimizer, step_size = 100, gamma = 0.1)
for (t in 1:kargs$iteration) {
  if (t==kargs$decay_iter){
    kargs$learning_rate=kargs$learning_rate*kargs$decay_alpha
  }
  
  ### -------- Forward pass -------- 
  
  y_pred_1 <- model_1(x_train_1)
  y_pred_2 <- model_2(x_train_2)
  y_pred_3 <- model_3(x_train_3)
 
  ### -------- compute loss -------- 
  
  ####### l2 loss of rest of params
  l2_loss=0
  # n_params_others=0
  for (i in seq(1,length(model_1$parameters),2)){
    # n_params_others=n_params_others+model_1$parameters[[i]]%>% length
    l2_loss=l2_loss+kargs$lamda_2 * sum(abs(model_1$parameters[[i]])**2)
  }
  for (i in seq(1,length(model_2$parameters),2)){
    # n_params_others=n_params_others+model_2$parameters[[i]]%>% length
    l2_loss=l2_loss+kargs$lamda_2 * sum(abs(model_2$parameters[[i]])**2)
  }
  for (i in seq(1,length(model_3$parameters),2)){
    # n_params_others=n_params_others+model_2$parameters[[i]]%>% length
    l2_loss=l2_loss+kargs$lamda_2 * sum(abs(model_3$parameters[[i]])**2)
  }
  
 
  ####### l1 loss of first layer
  ####### can manipulate 
  n_params_layer0=model_1$linear0 %>% length
  list_x=list(model_1$linear0,model_2$linear0,model_3$linear0)
  
  p=kargs$p
  if (post_train){
    p=0
  }
  l1_loss=kargs$lamda_1 * smooth_l1(list_x=list_x,p=p)
  
  
  ####### bce loss
  
  #y_pred=y_pred%>% torch_tensor(dtype = torch_float())
  
  bce_loss=nnf_binary_cross_entropy(y_pred_1,y_train_1)+
           nnf_binary_cross_entropy(y_pred_2,y_train_2)+
           nnf_binary_cross_entropy(y_pred_3,y_train_3)
  
  
  ####### total loss
  loss <- bce_loss+l1_loss+l2_loss
  

 
  if (t %% 30 == 0){
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  }
  

  ### -------- Backpropagation -------- 
  

  # compute gradient of the loss w.r.t. all learnable parameters of the model
  loss$backward()

  ### -------- Update weights -------- 
  
  # Wrap in with_no_grad() because this is a part we DON'T want to record
  # for automatic gradient computation
  # Update each parameter by its `grad`
  
  with_no_grad({
    model_1$parameters %>% purrr::walk(function(param) param$sub_(kargs$learning_rate * param$grad))
    model_1$linear0$sub_(kargs$learning_rate*model_1$linear0$grad)
    model_2$parameters %>% purrr::walk(function(param) param$sub_(kargs$learning_rate * param$grad))
    model_2$linear0$sub_(kargs$learning_rate*model_2$linear0$grad)
    model_3$parameters %>% purrr::walk(function(param) param$sub_(kargs$learning_rate * param$grad))
    model_3$linear0$sub_(kargs$learning_rate*model_3$linear0$grad)
    
    })
  
  # Zero the gradients before running the backward pass.
  model_1$zero_grad()
  model_2$zero_grad()
  model_3$zero_grad()

  model_1$linear0$grad$zero_()
  model_2$linear0$grad$zero_()
  model_3$linear0$grad$zero_()
  
  
  
}

  
  w_change_list=list(model_1$linear0,model_2$linear0,model_3$linear0)
  w_change_list[[1]]=as.numeric(w_change_list[[1]])
  w_change_list[[2]]=as.numeric(w_change_list[[2]])
  w_change_list[[3]]=as.numeric(w_change_list[[3]])
  
  nroot <- function(x,n){
    abs(x)^(1/n)*sign(x)
  }
  
  
  threshold=0.15
  mcl_w=NULL
  for(i in 1:length(model_1$linear0)){
    mcl_w[i]=nroot(as.numeric(model_1$linear0[i]*model_2$linear0[i]*model_3$linear0[i]),3)
  }
  max_mcl_w=max(abs(mcl_w))
  for(i in 1:length(model_1$linear0)){
    if (abs(nroot(as.numeric(w_change_list[[1]][i]*w_change_list[[2]][i]*w_change_list[[3]][i]),3))<=threshold*max_mcl_w){
      w_change_list[[1]][i]=w_change_list[[2]][i]=w_change_list[[3]][i]=0
    }else{
      w_change_list[[1]][i]=w_change_list[[1]][i]
      w_change_list[[2]][i]=w_change_list[[2]][i]
      w_change_list[[3]][i]=w_change_list[[3]][i]
    }
    
  }
  
  
  index=which(w_change_list[[1]]!=0)
  ans=NULL
  ans$model_list=list(model_1,model_2,model_3)
  ans$loss=bce_loss$item()
  ans$w_change_list=w_change_list
  ans$index=index
  
  return(ans)
}