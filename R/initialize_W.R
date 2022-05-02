.initialize_W = function(X, nW, Wseed = NULL) {
  ## Column-wise range of X 
  Xmin = apply(X, 2, min) 
  Xmax = apply(X, 2, max) 
  Xrange = Xmax - Xmin 
  #initmin = Xmin - 0.05*Xrange
  #initmax = Xmax + 0.05*Xrange 
  initmin = Xmin 
  initmax = Xmax
  
  ## Set seed, if given 
  if(!is.null(Wseed)) {
    set.seed(Wseed)
  }
  
  ## Sample W uniformly in [0,1], reset seed back to normal 
  W0 = matrix(runif(nW*ncol(X)), nrow = nW, ncol = ncol(X))
  set.seed(NULL)
  
  ## Scale W from [0,1] to [initmin, initmax] by dimension 
  for(d in 1:ncol(X)) {
    W0[,d] = (W0[,d] - 0)/(1 - 0)*(initmax[d] - initmin[d]) + initmin[d]
  }
  
  return(W0)
}