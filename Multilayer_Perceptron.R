library(pracma)
library(purrr)
library(zeallot)
library(caret)
library(dummies)
library(data.table)
library(purrr)
library(dplyr)

notas_alumnos <- fread('C:/Users/albgonzal/Documents/Desarrollo R/Algoritmos/data.csv', data.table = F, header = T)
notas_alumnos$`score-1` <- (notas_alumnos$`score-1` - min(notas_alumnos$`score-1`))/ (max(notas_alumnos$`score-1`) - min(notas_alumnos$`score-1`))
notas_alumnos$`score-2` <- (notas_alumnos$`score-2` - min(notas_alumnos$`score-2`))/ (max(notas_alumnos$`score-2`) - min(notas_alumnos$`score-2`))

for (i in seq(8)){ notas_alumnos<-rbind(notas_alumnos, notas_alumnos)}

####################################################################################################################################
#Creamos la clase "Layer" para crear cada capa de la red neuronal:

setClass("Layer",
         slots =  c(
           
           weights = "matrix",
           threshold = "matrix",
           out = "list",
           delta = "matrix",
           active_function = "list"
         ))

#Inicializamos los pesos y los valores umbrales de cada neurona para cada capa de forma aleatoria:

setMethod("initialize", signature("Layer"), function(.Object, n_con, n_neur, train, target, input_neuron = FALSE, active_function){
  
  
  #Definimos las posibles funciones de activación de las neuronas:
  
  sigmoid_fun<-list(sigmoid = function(x){return(1/(1+exp(-x)))}, 
                    sigmoid_deriv=function(x){x*(1-x)})
  
  tanh_fun<-list(tanh = function(x){return(tanh(x))},
                 tanh_deriv = function(x){1-(tanh(x)^2)})
  
  relu_fun<-list(relu = function(x){if_else(x<0, 0, 1)},
                 relu_deriv = function(x){if_else(x>0, 1, if_else(x==0, 0.5, 0))})
  
  softplus_fun <- list(sotfplus = function(x){log(1+exp(x))},
                       softplus_deriv = function(x){1/(1+exp(-x))})
  
  active_function <- switch(active_function,
                       sigmoid_fun = sigmoid_fun,
                       tanh_fun = tanh_fun,
                       relu_fun = relu_fun,
                       softplus_fun = softplus_fun)
  
  .Object@active_function = active_function
  
  .Object@weights<-rand(n_con, n_neur)
  
  f<-compose(partial(matrix, nrow = n_neur),
             partial(rep, times = dim(train[,-target])[1]),
             partial(rand, n_neur, 1))
  
  .Object@threshold<-f()
  
  if(input_neuron == T){
    
    .Object@out <- list(as.matrix(train[,-target]),as.matrix(train[,-target]))
  }
  
  return(.Object)
})
####################################################################################################################################

multilayer_perceptron<-function(train, test = NULL, target, topology, active_function, lr = 0.1, max_iter = 1000,
                                cost_function = "cost_fun", multiclass = FALSE, predict = FALSE){
  
  #Definimos una función para comprobar que los datos de entrada sean correctos:
  
  perceptron_comprobations<-function(train, test, target, topology, active_function, lr, max_iter, cost_function, multiclass, predict){

    if(!is.numeric(max_iter) || !is.numeric(lr) || !is.numeric(target)){stop("'lr', 'target' and 'max_iter' parameters must be numeric")}

    if(predict == F & !is.null(test)){stop("If 'predict' is false, then 'test' parameter must be FALSE")}

    if(predict != F & is.null(test)){stop("If 'predict' is TRUE, then 'test' must not be FALSE")}

    if(!is.numeric(topology) || !is.vector(topology)){stop("'topology' must be a numeric vector")}

    if(!is.data.frame(train) & !is.matrix(train)){stop("'train' must be a data frame or a matrix")}

    if(!is.data.frame(test) & !is.matrix(test) & !is.null(test)){stop("'test' must be a data frame, a matrix or null")}

    if(!is.logical(multiclass) || !is.logical(predict)){stop("'multiclass' and 'predict' parameters must be logical")}

    if(apply(train, 2, is.numeric) %>% sum() != dim(train)[2]){stop("'train' columns must be numeric")}

    if(!is.null(test)){

      if(apply(test, 2, is.numeric) %>% sum() != dim(test)[2]){stop("'test' columns must be numeric")}

    }
    
    active_function<-match.arg(active_function,
                               c("sigmoid_fun", "tanh_fun", "relu_fun", "softplus_fun"),
                               several.ok = TRUE)
    
    cost_function<-match.arg(cost_function,
                             c("cost_fun"),
                             several.ok = FALSE)
    
    if(length(topology) != length(active_function)){stop("The length of 'active_function' and 'topology' must be the same. The values of
                                                         'active_function' has to be contained in 'sigmoid_fun', 'relu_fun', 'softplus_fun'
                                                         and/or 'tanh_fun'")}

  }
  
  #Llamamos a la función de la comprobación de parámetros:
  
  do.call(perceptron_comprobations, list(train, test, target, topology, active_function, lr, max_iter, cost_function, multiclass, predict))
  
  #Definimos la función de coste:
  
  cost_fun<-list(cost = function(pred, real){return(mean((pred-real)^2))},
                 cost_deriv = function(pred, real){return(pred-real)})
  
  
  #Definimos la función para crear la estructura del perceptrón multicapa:
  
  create_perceptron <- function(topology, train, target, active_function){
    
    neural_net<-map(seq(length(topology)), function(i){
      
      if(i == 1){
        
        new("Layer", n_con = 1, n_neur = 1, 
            train = train, 
            target = target, 
            input_neuron = TRUE, 
            active_function = active_function[i])
        
        
      } else{
          
          new("Layer",
              n_con = topology[i-1],
              n_neur = topology[i], 
              train = train, 
              target = target, 
              input_neuron = FALSE, 
              active_function = active_function[i])

      }
      
    })
    
    names(neural_net)<-map_chr(seq(length(topology)), function(i){
      
      paste("Layer_", i, sep = "")
      
    })
    
    return(neural_net)
    
  }
  
  #Definimos la función que aplica el algoritmo forward_propagation:
  
  forward_propagation <- function(neural_net){
    
    
    for (layer in seq_along(neural_net)[-1]) {
      
      z<-as.matrix(neural_net[[layer-1]]@out[[2]]) %*% neural_net[[layer]]@weights + t(neural_net[[layer]]@threshold)
      
      neural_net[[layer]]@out <- list(z, neural_net[[layer]]@active_function[[1]](z))
      
    }
    
    return(neural_net)
    
  }
  
  
  #Definimos la función que aplica el algoritmo back_propagation:
  
  back_propagation<-function(neural_net, x_test, cost_function){
    
    x_test<-as.matrix(x_test)
    
    for (i in order(seq(length(neural_net)), decreasing = T)) {
      
      if(i == length(neural_net)){
        
        neural_net[[i]]@delta <- (cost_fun[[2]](neural_net[[i]]@out[[2]], x_test) * neural_net[[i]]@active_function[[2]](neural_net[[i]]@out[[2]])) 
        
      } else {
        
        neural_net[[i]]@delta <- (neural_net[[i+1]]@delta %*% t(neural_net[[i+1]]@weights)) * neural_net[[i]]@active_function[[2]](neural_net[[i]]@out[[2]])
        
      }
      
    }
    
    return(neural_net)
    
    
  }
  
  
  #Definimos la función que aplica el algoritmo del descenso del gradiente para entrenar la red:
  
  gradient_descent_perceptron<-function(neural_net, x_test, lr, max_iter, threshold = NULL){
    
    i<-1
    
    while(i < max_iter){
      
      for (j in c(2:length(neural_net))) {
        
        neural_net[[j]]@threshold <- neural_net[[j]]@threshold - (lr * t(neural_net[[j]]@delta))
        
        #neural_net[[j]]@weights <- neural_net[[j]]@weights - lr * t(neural_net[[j-1]]@out[[2]]) %*% (neural_net[[j]]@delta)  
        
        neural_net[[j]]@weights <- neural_net[[j]]@weights - lr * mean(neural_net[[j]]@delta)*mean(neural_net[[j-1]]@out[[2]]) 
        
      }
      
      neural_net<-forward_propagation(neural_net = neural_net) %>%
        back_propagation(., x_test, cost_function = cost_function)
      
      i<-i+1
      
    }
    
    
    
    return(neural_net)
    
  }
  
  #Actualizamos las dimensiones de threshold para poder predecir con el test y actualizamos la salida de la primera capa:
  
  perceptron_update<-function(neural_net, test, target){
    
    for (i in seq(length(neural_net))) {
      
      if(dim(neural_net[[i]]@threshold)[1] == 1){
        
        slot(neural_net[[i]], "threshold") <- slot(neural_net[[i]], "threshold")[,seq(test[,target])] %>% t() %>% as.matrix()
        
      } else {
        
        slot(neural_net[[i]], "threshold") <- slot(neural_net[[i]], "threshold")[,seq(test[,target])]
        
      }
      
      if(i == 1){
        
        slot(neural_net[[i]], "out") <- list(as.matrix(test[, -target]), as.matrix(test[, -target])) 
        
      }
      
    }
    
    return(neural_net)
    
  }
  
  
  #Seleccionamos la función de coste según el parámetro seleccionado:
  
  cost_function<-switch(cost_function,
                        cost_fun = cost_fun)
  
  
  #Componemos todas las funciones para crear y entrenar el perceptrón:
  
  perceptron<-compose(forward_propagation,
                      partial(gradient_descent_perceptron, x_test = train[, target], lr = lr, max_iter = max_iter),
                      partial(back_propagation, x_test = train[, target],cost_function = cost_function),
                      forward_propagation,
                      partial(create_perceptron, topology = topology, train = train, target = target, active_function = active_function)
  )
  
  
  #Realizamos todo el proceso:
  
  perceptron_multilayer<-perceptron()
  
  
  if(predict == TRUE){
    
    perceptron_multilayer<-perceptron_update(perceptron_multilayer, test, target) %>% forward_propagation()
    
  }
  
  
  #Si las clases a predecir son mayores que 2, vamos por este camino (deben introducirse en la variable 'y' como dummies)
  
  if(multiclass == TRUE){
    
    c(class_levels, values_levels) %<-% list(names(train[,target]), seq(dim(train[,target])[2]))
    
    perceptron_multilayer[[length(perceptron_multilayer)]]@out[[3]]<-apply(perceptron_multilayer[[length(perceptron_multilayer)]]@out[[2]], 1, which.max)
    
    return(perceptron_multilayer)
    
    #Si las clases son binarias, van por el 'else'
    
    
  } else {
    
    pred<-ifelse(perceptron_multilayer[[length(perceptron_multilayer)]]@out[[2]] > 0.5, 1, 0)
    
    if(predict == TRUE){
      
      conf_matrix <- confusionMatrix(as.factor(pred), as.factor(test[,target]))
      
    } else {
      
      conf_matrix <- confusionMatrix(as.factor(pred), as.factor(train[,target]))
      
    }
    
    
    return(list(perceptron_multilayer = perceptron_multilayer,
                conf_matrix = conf_matrix))
    
  }
  
  
}




















