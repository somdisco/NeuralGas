#' Neural Gas Batch Learning 
#' 
#' @param X a data matrix, one observation per row. 
#' 
#' @param W the number of NG prototypes and their initialization, see \code{\link{NGWInitialization}}.  
#' 
#' @param lambda0 the starting value of the neighborhood factor for cooperative learning. 
#' Typical values are 25-50\% of the number of prototypes in the network. Optional, default = 0.25*nrow(W).
#' 
#' @param lambda_decay a multiplicative decay factor applied to lambda after every learning epoch. Ex.: lambda(t) = lambda(t-1)*decay.
#' Optional, default = 0.9. Must be < 1 to ensure convergence.  Higher values require longer training time, but typically give better quantization. 
#' 
#' @param lambda_schedule a named vector to set a custom annealing schedule for lambda instead of the multiplicative decay controlled by \code{lambda} and \code{lambda_decay}. 
#' \code{names(lambda_schedule)} should be integers defining the epoch \strong{through} which the corresponding elements of the vector are applied. 
#' Optional; if given, this schedule over-rides any multiplicative annealing specified by \code{lambda} and \code{lambda_decay}.  
#' 
#' @param tol_delBMU tolerance controlling convergence of the learning, see \code{\link{NGConvergence}}. 
#' Optional, default = 1. 
#' 
#' @param tol_delMQE tolerance controlling convergence of the learning, see \code{\link{NGConvergence}}. 
#' Optional, default = 0.1. 
#' 
#' @param max_epochs tolerance controlling convergence of the learning, see \code{\link{NGConvergence}}. 
#' Optional, default = 999999. 
#' 
#' @param XL optionally, a vector of labels for the data in the rows of X.  These can be of any type (character, factor, numeric) but will be coerced to strings internally. 
#' If given, XL allows reporting additional quality measures during the learning process as described in \code{\link{NGLearnHist}}.  
#' 
#' @param parallel whether to compute in parallel (recommended, default = TRUE). 
#' @param verbose whether to print learning history to the console after each epoch. Default = TRUE. 
#' 
#' @return a list with components: 
#' \describe{
#' \item{W}{the learned prototype matrix, one prototype per row}
#' \item{epochs}{the number of epochs trained}
#' \item{lambda_start}{the value of lambda at the beginning of learning (the value of initial lambda given)}
#' \item{lambda_end}{the value of lambda at the end of leanring}
#' \item{lambda_decay}{the supplied decay factor}
#' \item{lambda_schedule}{the schedule set in \code{lambda_schedule}, if given}
#' \item{tol_delBMU}{the supplied value of tol_delBMU}
#' \item{tol_delMQE}{the supplied value of tol_delMQE}
#' \item{max_epochs}{the supplied value of max_epochs}
#' \item{exec_time}{execution time, in minutes}
#' \item{convergence}{whether the network achieved the \code{tol_delBMU} and \code{tol_delMQE} convergence tolerances}
#' \item{LearnHist}{a data frame recording various learning histories, see \code{\link{NGLearnHist}}}
#' }
#' 
#' @details 
#' Neural gas finds prototypes \code{W} which minimize the following cost function: 
#' \deqn{\sum_i \sum_j h_{ij} d(x_i, w_j)}{sum_i(sum_j( h_ij x d(x_i,w_j) ))}
#' where the neighborhood function 
#' \deqn{h_{ij} = exp(-k_{ij} / \lambda)}{h_ij = exp(-k_ij / lambda)}
#' and \eqn{k_{ij}}{k_ij} = the rank of \eqn{d(x_i, w_j)}, with respect to all other \eqn{d(x_i, w_k)}.    
#' Ranks are ascending, and by convention start at 0 (instead of 1). 
#'
#' Batch learning updates the prototypes after presentation of all data to the network. 
#' The update rule is implemented according to the method of:
#' Cottrell, M., Hammer, B., Hasenfuss, A., & Villmann, T. (2006). \emph{Batch and median neural gas}
#' 
#'
#' 
#' @export
NGBatch = function(X, W, lambda0 = 0.25*nrow(W), lambda_decay = 0.9, lambda_schedule = NULL, 
                   tol_delBMU = 1, tol_delMQE = 0.1, max_epochs = 999999, 
                   XL = NULL, parallel = TRUE, verbose = TRUE, 
                   dist = "L2", MHL = NULL) {
  
  
  ## Decode the input for W. 
  # If numel(W)=1, assume W(0) = # of prototypes 
  # If numel(W)=2, assume W(0) = # of prototypes, W(1) is a random seed 
  # If ncols(W)=ncols(X), assume W is an initial prototype matrix, and just scale it to [0,1] using X's range 
  if(length(W)==1) {
    W = .initialize_W(X = X, nW = W[1])
  } else if(length(W)==2) {
    W = .initialize_W(X = X, nW = W[1], Wseed = W[2])
  } else if(ncol(W) != ncol(X)) {
    stop("Cannot decode W");
  } 
  
 
  ## Check on labels. 
  # If given, factorize and convert to integers 
  if(!is.null(XL)) {
    XL = as.character(XL)
  }
  
  if(!is.null(lambda_schedule)) {
    ## Make sure it is a named vector with integer names 
    epochs = suppressWarnings(as.integer(names(lambda_schedule)))
    if(any(is.na(epochs))) stop("lambda_schedule must be a named vector with integer names")
    if(any(diff(epochs) < 0)) stop("names(lambda_schedule) must be non-decreasing")
  }
  
  ## Check distance 
  if(!(dist == "L2" || dist == "Mahalanobis")) stop("dist must be either 'L2' or 'Mahalanobis'")
  if(dist == "Mahalanobis" && is.null(MHL)) stop("if dist='Mahalanobis', MHL must be given")
  if(dist == "Mahalanobis") {
    if(is.vector(MHL) && length(MHL) == ncol(X)) {
      MHL_diag = MHL
      MHL_full = NULL 
    } else if(is.matrix(MHL) && length(as.vector(MHL)) == ncol(X)) {
      MHL_diag = as.vector(MHL)
      MHL_full = NULL
    } else if(is.matrix(MHL) && nrow(MHL) == ncol(X) && ncol(MHL) == ncol(X)) {
      MHL_diag = NULL 
      MHL_full = MHL 
    } else 
      stop("Cannot decode MHL, check its dimensions.")
  } else {
    MHL_diag = NULL
    MHL_full = NULL 
  }
  

  ## Call cpp learn function 
  out = NeuralGas:::.cpp_NGLearn_batch(X = X, W = W, lambda0 = lambda0, lambda_decay = lambda_decay, lambda_schedule = lambda_schedule, 
                                       tol_delBMU = tol_delBMU, tol_delMQE = tol_delMQE, max_epochs = max_epochs, XL = XL, 
                                       parallel = parallel, verbose = verbose, 
                                       dist = dist, MHLdiag = MHL_diag) # only support diagonal Mahal distances for now
  

  return(out)
}






#' Neural Gas Online Learning 
#' 
#' @param X a data matrix, one observation per row. 
#' @param W the number of NG prototypes and their initialization, see \code{\link{NGWInitialization}}.  
#' 
#' @param alpha0 the starting value of the learning rate used for prototype updates. Optional, default = 0.5. 
#' @param alpha_decay a multiplicative decay factor applied to \code{alpha} after every iteration. Ex.: alpha(t) = alpha(t-1)*decay.
#' Optional, default = 0.9^(1/\code{nrow(X)}). 
#' @param alpha_schedule a named vector to set a custom annealing schedule for alpha instead of the multiplicative decay controlled by \code{alpha} and \code{alpha_decay}. 
#' \code{names(alpha_schedule)} should be integers defining the epoch \strong{through} which the corresponding elements of the vector are applied. 
#' Optional; if given, this schedule over-rides any multiplicative annealing specified by \code{alpha} and \code{alpha_decay}.  
#' 
#' @param lambda0 the starting value of the neighborhood factor for cooperative learning. 
#' Typical values are 25-50\% of the number of prototypes in the network. Optional, default = 0.25*nrow(W).
#' @param lambda_decay a multiplicative decay factor applied to lambda after every learning epoch. Ex.: lambda(t) = lambda(t-1)*decay.
#' Optional, default = 0.9^(1/nrow(X)). 
#' @param lambda_schedule a named vector defining an annealing schedule for \code{lambda}, in the same format as \code{alpha_schedule}. 
#' Mandatory if \code{alpha_schedule} is set. 
#' 
#' @param tol_delBMU tolerance controlling convergence of the learning, see \code{\link{NGConvergence}}. Optional, default = 1. 
#' @param tol_delMQE tolerance controlling convergence of the learning, see \code{\link{NGConvergence}}. Optional, default = 0.1. 
#' @param max_epochs tolerance controlling convergence of the learning, see \code{\link{NGConvergence}}. Optional, default = 999999. 
#' 
#' @param XL optionally, a vector of labels for the data in the rows of X.  These can be of any type (character, factor, numeric) but will be coerced to strings internally. 
#' If given, Xlabel allows reporting additional quality measures during the learning process as described in \code{\link{NGLearnHist}}.  
#' 
#' @param parallel whether to compute in parallel (recommended, default = TRUE). 
#' @param verbose whether to print learning history to the console after each epoch. Default = TRUE. 
#' 
#' @param Xseed the seed value controlling the random sampling of X for presentation to the network at each iteration. 
#' Optional, default = NULL does not set this random seed. 
#' 
#' @return a list with components: 
#' \describe{
#' \item{W}{the learned prototype matrix, one prototype per row}
#' \item{iterations}{the number of iterations trained}
#' \item{epochs}{the number of epochs (= iterations / nrow(X)) trained}
#' \item{alpha_start}{the value of alpha at the beginning of learning (the value of initial lambda given)}
#' \item{alpha_end}{the value of alpha at the end of learning}
#' \item{alpha_decay}{the supplied decay factor}
#' \item{alpha_schedule}{the schedule set in \code{alpha_schedule}, if given}
#' \item{lambda_start}{the value of lambda at the beginning of learning (the value of initial lambda given)}
#' \item{lambda_end}{the value of lambda at the end of leanring}
#' \item{lambda_decay}{the supplied decay factor}
#' \item{lambda_schedule}{the schedule set in \code{lambda_schedule}, if given}
#' \item{tol_delBMU}{the supplied value of tol_delBMU}
#' \item{tol_delMQE}{the supplied value of tol_delMQE}
#' \item{max_epochs}{the supplied value of max_epochs}
#' \item{exec_time}{execution time, in minutes}
#' \item{convergence}{whether the network achieved the \code{tol_delBMU} and \code{tol_delMQE} convergence tolerances}
#' \item{LearnHist}{a data frame recording various learning histories, see \code{\link{NGLearnHist}}}
#' }
#' 
#' @details 
#' Neural gas finds prototypes \code{W} which minimize the following cost function: 
#' \deqn{\sum_i \sum_j h_{ij} d(x_i, w_j)}{sum_i(sum_j( h_ij x d(x_i,w_j) ))}
#' where the neighborhood function 
#' \deqn{h_{ij} = exp(-k_{ij} / \lambda)}{h_ij = exp(-k_ij / lambda)}
#' and \eqn{k_{ij}}{k_ij} = the rank of \eqn{d(x_i, w_j)}, with respect to all other \eqn{d(x_i, w_k)}.  
#' Ranks are ascending, and by convention start at 0 (instead of 1). 
#'
#' Online learning updates the prototypes after presentation of a single datum to the network (one learning iteration). 
#' The update rule is 
#' \deqn{w(t+1) = w(t) + \alpha  h_{ij}  (x_i - w_j)}{w(t+1) = w(t) + alpha* h_ij*(x_i - w_j)}
#' 
#' @export
NGOnline = function(X, W, 
                    alpha0 = 0.5, alpha_decay = 0.9^(1/nrow(X)), alpha_schedule = NULL, 
                    lambda0 = 0.25*nrow(W), lambda_decay = 0.9^(1/nrow(X)), lambda_schedule = NULL, 
                    tol_delBMU = 1, tol_delMQE = 0.1, max_epochs = 999999, 
                    XL=NULL, parallel = TRUE, verbose = TRUE, Xseed = NULL) {
  
  ## Decode the input for W. 
  # If numel(W)=1, assume W(0) = # of prototypes 
  # If numel(W)=2, assume W(0) = # of prototypes, W(1) is a random seed 
  # If ncols(W)=ncols(X), assume W is an initial prototype matrix, and just scale it to [0,1] using X's range 
  if(length(W)==1) {
    W = .initialize_W(X = X, nW = W[1])
  } else if(length(W)==2) {
    W = .initialize_W(X = X, nW = W[1], Wseed = W[2])
  } else if(ncol(W) != ncol(X)) {
    stop("Cannot decode W");
  } 
  
  ## Reset random seed 
  set.seed(Xseed)
  
  ## Check on labels. 
  # If given, factorize and convert to integers 
  if(!is.null(XL)) {
    XL = as.character(XL)
  }
  
  
  if(!is.null(alpha_schedule) && !is.null(lambda_schedule)) {
    ## Make sure it is a named vector with integer names 
    lambda_epochs = suppressWarnings(as.integer(names(lambda_decay)))
    if(any(is.na(lambda_epochs))) stop("lambda_schedule must be a named vector with integer names")
    if(any(diff(lambda_epochs) < 0)) stop("names(lambda_schedule) must be non-decreasing")
    alpha_epochs = suppressWarnings(as.integer(names(alpha_decay)))
    if(any(is.na(alpha_epochs))) stop("alpha_schedule must be a named vector with integer names")
    if(any(diff(alpha_epochs) < 0)) stop("names(alpha_schedule) must be non-decreasing")
  }
  
  
  ## Call cpp learn function 
  out = NeuralGas:::.cpp_NGLearn_online(X = X, W = W, 
                                        alpha0=alpha0, alpha_decay=alpha_decay, alpha_schedule = alpha_schedule, 
                                        lambda0 = lambda0, lambda_decay = lambda_decay, lambda_schedule = lambda_schedule, 
                                        tol_delBMU = tol_delBMU, tol_delMQE = tol_delMQE, max_epochs = max_epochs, XL = XL, 
                                        parallel = parallel, verbose = verbose)
  
  # if(is.null(alpha_schedule) && is.null(lambda_schedule)) {
  #   ## Call cpp learn function 
  #   out = NeuralGas:::.cpp_NGLearn_online(X = X, W = W, 
  #                                         alpha0=alpha0, alpha_decay=alpha_decay, lambda0 = lambda0, lambda_decay = lambda_decay, 
  #                                         tol_delBMU = tol_delBMU, tol_delQE = tol_delQE, max_epochs = max_epochs, XLabel = XLabel, 
  #                                         parallel = parallel, verbose = verbose)
  # } else if(!is.null(alpha_schedule) && !is.null(lambda_schedule)) {
  #   
  #   ## Make sure it is a named vector with integer names 
  #   lambda_epochs = suppressWarnings(as.integer(names(lambda_decay)))
  #   if(any(is.na(lambda_epochs))) stop("lambda_schedule must be a named vector with integer names")
  #   if(any(diff(lambda_epochs) < 0)) stop("names(lambda_schedule) must be non-decreasing")
  #   alpha_epochs = suppressWarnings(as.integer(names(alpha_decay)))
  #   if(any(is.na(alpha_epochs))) stop("alpha_schedule must be a named vector with integer names")
  #   if(any(diff(alpha_epochs) < 0)) stop("names(alpha_schedule) must be non-decreasing")
  #   
  #   ## Call cpp learn function 
  #   out = NeuralGas:::.cpp_NGLearn_online(X = X, W = W, 
  #                                         alpha_schedule=alpha_schedule, lambda_schedule = lambda_schedule, 
  #                                         tol_delBMU = tol_delBMU, tol_delQE = tol_delQE, max_epochs = max_epochs, XLabel = XLabel, 
  #                                         parallel = parallel, verbose = verbose)
  # } else {
  #   stop("alpha_schedule and lambda_schedule must both me (1) scalars or (2) a named vector giving an annealing schedule")
  # }
  # 
  
  return(out)
}





#### ***** Documentation for Learning History 
#' Learning History for NeuralGas Objects
#' 
#' \strong{Note:} This is not a function, merely a description of the \code{LearnHist} data frame returned from the \code{NGBatch} and \code{NGOnline} functions. 
#' 
#' At the end of each epoch the following monitoring measures are computed, reported, and stored in the \code{LearnHist} data frame 
#' returned in the output list. For online learning one 'Epoch' is equivalent to \code{N} learning iterations, where \code{N} is the 
#' number of training vectors:  \code{N = nrow(X)} where \code{X} is the matrix of training data.  
#' 
#' \describe{
#' \item{Epoch}{the epoch for which measures are reported}
#' \item{alpha}{the effective learning rate}
#' \item{lambda}{the effective neighborhood lambda}
#' \item{Cost}{the value of the NG cost function, divided by the number of data vectors \code{N}. 
#' The purpose of division by \code{N} is to put \code{Cost} on a similar scale to \code{MQE}.}
#' \item{MQE}{Mean Quantization Error of all data}
#' \item{NhbEff}{the effect of the current value of lambda on the network, calculated as \code{Cost} / \code{MQE}. 
#' \code{NhbEff} is always >= 1, with values = 1 indicating no neighborhood effect, which occurs as lambda -> 0.}
#' \item{delCost}{the relative absolute percent change in Cost, from epoch(t-1) to epoch(t), = 
#' abs(Cost(t) - Cost(t-1)) / Cost(t) * 100 }
#' \item{delMQE}{the relative absolute percentage change in the MQE, from epoch(t-1) to epoch(t).}
#' \item{delBMU}{the proportion of data whose BMU has changed from epoch(t-1) to epoch(t)}
#' \item{Entropy}{Normalized Shannon Entropy of the VQ mapping at epoch(t)}
#' }
#' 
#' If data labels are given (by supplying a value for \code{Xlabel}) we can use the learned VQ mapping to project these labels onto 
#' the prototypes. Each prototype's label (denoted RFL = Receptive Field Label) is decided by plurality vote of the labels of the data in its receptive field (RF). 
#' In the presence of labels, additional measures are computed, reported and stored in \code{LearnHist}: 
#' \describe{
#' \item{PurityWOA}{Weighted Overall Average of the individual Purity scores of each receptive field. 
#' The Purity score of each RF = \#(label(x)==RFL)/\#(RF), for all data x in the RF. This measures how much label confusion 
#' exists in the RF; ideally, if the labels indicate well separated classes / clusters we would have Purity=1. The value reported in 
#' \code{PurityWOA} is the average Purity score of each RF, weighted by the RF's size (number of data vectors mapped to it). 
#' Purity -> 0 as intra-RF label confusion increases.}
#' \item{WLUnq}{The number of unique prototype labels. This is a helpful measure to determine how well the prototypes represent X 
#' in situations with unbalanced class size, particularly when there are rare classes (of small size). Ideally, all unique labels 
#' found in XL should be represented by some (set of) prototype(s).}
#' }
#' \item{WLHell}{The Hellinger Distance between the empirical categorical distributions of XL and WL.  
#' WLHell=0 means the distributions perfectly align; any value > 0 indicates disagreement. 
#' For example, assume the data are labeled one of A, B, or C, and (empirically, according to XL), 
#' pX(A) = 0.20, pX(B) = 0.30 and pX(C) = 0.50. If the VQ has produced the mapping pRF(A) = 0.20, pRF(B) = 0.30 and pRF(C) = 0.50, then 
#' the distributions of data and RF labels perfectly agree, and WLHell = 0. 
#' }
#' 
#' @name NGLearnHist
NULL 


### ***** Documentation for Convergence 
#' Convergence for NeuralGas Objects 
#' 
#' \strong{Note:} This is not a function, merely a description of the convergence criteria used in the \code{NGBatch} and \code{NGOnline} functions. 
#' 
#' NeuralGas learning is terminated by the first occurrence of the following convergence criteria reaching their user-supplied tolerances: 
#' 
#' \itemize{
#' \item The training age (number of learning epochs performed) exceeding the parameter \code{max_epochs} 
#' 
#' \item The \code{tol_delBMU} AND \code{tol_delMQE} tolerances being met for three consecutive epochs. 
#' These criteria, described below, measure the stability of the vector quantization over training time.  
#' }
#' 
#' \code{delBMU} reports the percentage of training vectors whose BMU has changed from one epoch to the next. 
#' For example, \code{delBMU < tol_delBMU = 1} means that 
#' less than 1\% of the data have changed their BMU from epoch \code{t-1} to \code{t}. 
#' 
#' \code{delMQE} reports the average (absolute) percentage change in Mean Quantization Error (MQE) from one epoch to the next. 
#' 
#' Tto train for a fixed number of epochs \code{E}, set \code{max_epochs = E} and \code{tol_delBMU = tol_delMQE = 0}. 
#' 
#' For online learning an epoch is equivalent to \code{N} learning iterations, where \code{N} = number of training vectors. 
#' 
#' During training early termination can be achieved by user interruption (typing CTRL-C in the R console window).  
#' Upon detecting early termination, both \code{NGBatch} and \code{NGOnline} will return the current state of the network 
#' (existing values of the prototypes and any training history logged).  
#' 
#' 
#' @name NGConvergence
NULL 


### ***** Documentation for initial W 
#' Prototype Initialization for NeuralGas Objects 
#' 
#' \strong{Note:} This is not a function, merely a description of the types of prototype initialization understood by the \code{NGBatch} and \code{NGOnline} functions. 
#' 
#' Both \code{NGBatch} and \code{NGOnline} require the input parameter \code{W} which describes both 
#' the number of prototypes in the network, and how they are initialized.  As such, the value supplied for \code{W} can take 
#' one of the three following forms: 
#' 
#' \describe{
#' \item{a single number}{giving the number of prototypes in the network. 
#' In this case, prototypes are initialized randomly and uniformly in the range of training data \code{X}.}
#' 
#' \item{a length=2 vector}{giving the number of prototypes (first element) and the random seed used to initialize them (second element). 
#' Use this option for random prototype initialization with a fixed seed for reproducibility.}
#' 
#' \item{a matrix}{giving the initial prototypes in its rows. ncols(W) should = ncol(X), and the number of prototypes in the network is set equal to \code{nrow(W)}
#' If initial prototypes are given they should occupy the same range as \code{X}, 
#' as they will be linearly scaled internally from \[max(X),min(X)\] to \[0,1\] prior to learning.}
#' }
#' 
#' @name NGWInitialization
NULL 

