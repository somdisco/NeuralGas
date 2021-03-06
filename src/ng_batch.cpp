#ifndef NEURALGAS_HPP
#include "neuralgas.hpp"
#endif


struct update_batch_prototypes_prlwkr : public RcppParallel::Worker {
  
  // Inputs 
  const arma::mat& X;
  double minX, maxX; 
  arma::mat& W; 
  const double& lambda; 
  
  // Intermediary variables 
  unsigned int nX, nW, d; 
  std::string dist; 
  arma::rowvec MHLdiag; 
  
  // Accumulated results
  arma::mat sum_hX; // nW x d matrix whose rows contain sum(h*x) for each prototype 
  arma::vec sum_h;  // length nW vector whose elements contain sum(h) for each prototype 
  arma::uvec BMUs;   // BMU for each data point 
  arma::vec QEs;     // QE for each data point 
  arma::vec Costs; 
  
  
  // constructors
  update_batch_prototypes_prlwkr(const arma::mat& X, double minX, double maxX, arma::mat& W, const double& lambda) : 
    X(X), minX(minX), maxX(maxX), W(W), lambda(lambda), 
    nX(X.n_rows), nW(W.n_rows), d(X.n_cols), 
    sum_hX(arma::zeros<arma::mat>(nW,d)), sum_h(arma::zeros<arma::vec>(nW)), 
    BMUs(arma::zeros<arma::uvec>(nX)), QEs(arma::zeros<arma::vec>(nX)), Costs(arma::zeros<arma::vec>(nX)) {
    // Set default distance 
    dist = "L2";
  };
  
  
  update_batch_prototypes_prlwkr(const update_batch_prototypes_prlwkr& me, RcppParallel::Split) :
    X(me.X), minX(me.minX), maxX(me.maxX), W(me.W), lambda(me.lambda), 
    nX(me.nX), nW(me.nW), d(me.d), dist(me.dist), MHLdiag(me.MHLdiag), 
    sum_hX(arma::zeros<arma::mat>(nW,d)), sum_h(arma::zeros<arma::vec>(nW)),
    BMUs(arma::zeros<arma::uvec>(nX)), QEs(arma::zeros<arma::vec>(nX)), Costs(arma::zeros<arma::vec>(nX)) {};
  

  // Set the optional Mahalanobis distance 
  void set_Mahalanobis_diag(arma::rowvec m_diag) {
    if(!arma::all(m_diag > 0)) Rcpp::stop("Mahalanobis distance must have positive diagonal");
    if(m_diag.n_elem != X.n_cols) Rcpp::stop("Mahalanobis distance has incorrect size");
    dist = "Mahalanobis_diag";
    MHLdiag = m_diag; 
  }
  
  // Function to apply update from a single x
  void update_from_xi(unsigned int i) {
    
    // Scale this data vector from [min,max] to [0,1]. 
    // Do the scaling here to prevent having to store a scaled copy of X in memory 
    arma::rowvec x = (X.row(i) - minX) / (maxX - minX);
    
    // Rank the prototypes
    arma::uvec proto_rank;
    arma::uvec tmpBMU; arma::vec tmpQE; 
    arma::vec dW; 
    if(dist == "L2") {
      //rank_prototypes(proto_rank, BMUs(i), QEs(i), dW, x, W);
      dW = dist_to_protos_L2(x, W);
      rank_prototypes(proto_rank, BMUs(i), QEs(i), dW);
    } else if(dist == "Mahalanobis_diag") {
      //rank_prototypes_mahalanobis_diag(proto_rank, BMUs(i), QEs(i), dW, x, W, mahal_diag);
      dW = dist_to_protos_Mahalanobis_diag(x, W, MHLdiag);
      rank_prototypes(proto_rank, BMUs(i), QEs(i), dW);
    }
    

    // Compute the neighborhood factor
    arma::vec h = arma::exp(-arma::conv_to<arma::vec>::from(proto_rank) / lambda);
    
    // Compute matrix of h*x
    arma::mat hX(nW, d);
    hX.each_row() = x; 
    hX.each_col() %= h;
    
    // add these to running totals
    sum_hX += hX;
    sum_h += h;
    
    // compute cost 
    Costs(i) = arma::accu(h % dW); 
  }
  
  
  // process a block of x
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++) {
      update_from_xi(i);
    }
  }
  
  
  // join my values with that of another thread
  void join(const update_batch_prototypes_prlwkr& rhs) {
    sum_hX += rhs.sum_hX;
    sum_h += rhs.sum_h;
    BMUs += rhs.BMUs;
    QEs += rhs.QEs;
    Costs += rhs.Costs; 
  }
  
  
  void calc_parallel() {
    RcppParallel::parallelReduce(0, nX, *this);
  }
  
  
  void calc_serial() {
    for(unsigned int i=0; i<nX; ++i) {
      update_from_xi(i);
    }
  }
  
  void update_W() {
    //arma::mat tmpW = sum_hX;
    //tmpW.each_col() /= sum_h;
    //return tmpW;
    
    W = sum_hX; 
    W.each_col() /= sum_h; 
  }
  
  // Function to clear the computational results from the last call 
  void clear() {
    sum_hX.zeros(); 
    sum_h.zeros(); 
    BMUs.zeros(); 
    QEs.zeros(); 
    Costs.zeros(); 
  }
};


// [[Rcpp::export(".cpp_NGLearn_batch")]]
Rcpp::List cpp_NGLearn_batch(const arma::mat& X, arma::mat W, 
                             double tol_delBMU = 0.0, double tol_delMQE = 0.0, int max_epochs = 999999, 
                             double lambda0 = 0.0, double lambda_decay = 0.0, Rcpp::Nullable<Rcpp::NumericVector> lambda_schedule = R_NilValue, 
                             Rcpp::Nullable<std::vector<std::string>> XL = R_NilValue, 
                             bool parallel = true, bool verbose = true, 
                             std::string dist = "L2", 
                             Rcpp::Nullable<Rcpp::NumericVector> MHLdiag = R_NilValue) {
  // Inputs: 
  // X = data matrix, in external network range
  // minX,maxX = limits of external network range; now compute this internally  
  // W = prototype matrix, in internal network range [0,1] 
  // lambda = starting lambda value for training 
  // decay = decay rate, annealing done at every epoch according to lambda = lambda * decay
  
  
  // *** Initialize containers related to learning 
  double minX = X.min();        // external network range
  double maxX = X.max(); 
  double lambda = lambda0;      // initial neighborhood  
  arma::uvec rank(W.n_rows);    // the prototype rank @ each iter 
  unsigned int age = 0;         // num of learning epochs performed so far (epoch = nX iters, nX = nrow(X))
  
  // *** Scale W to [0,1] using min/max of X 
  W = (W - minX) / (maxX - minX); 
  
  // *** Initialize learning rate annealer 
  NG_learnrate_annealer annealer; 
  if(lambda_schedule.isNotNull()) {
    Rcpp::NumericVector lambda_schedule_ = Rcpp::as<Rcpp::NumericVector>(lambda_schedule); 
    annealer.set_scheduled_annealing(lambda_schedule_); 
  } else if(lambda > 0.0 && lambda_decay > 0.0) {
    annealer.set_multiplicative_annealing(lambda, lambda_decay); 
  } else {
    Rcpp::stop("Cannot parse lambda annealing. Input either (1) positive values for lambda & lambda_decay, or (2) a lambda_schedule");
  }
  
  
  // *** Initialize containers related to monitoring & reporting 
  unsigned int npass_delBMU = 0;  // num of consecutive monitorings the delBMU tol has been met 
  unsigned int npass_delMQE = 0;   // num of consecutive monitorings the delQE tol has been met 
  bool exit_flag = false;         // whether all conv criteria have been met 
  
  arma::uvec BMUs(X.n_rows); 
  BMUs.fill(std::numeric_limits<unsigned int>::quiet_NaN()); 
  arma::vec QEs(X.n_rows);         // QE of all data @ current monitoring 
  QEs.fill(std::numeric_limits<double>::quiet_NaN()); 
  arma::vec Costs(X.n_rows);         // QE of all data @ current monitoring 
  Costs.fill(std::numeric_limits<double>::quiet_NaN()); 
  
  NG_deltas deltas;
  deltas.update(BMUs, QEs, Costs);
  
  // Learn history storage 
  NG_learnhist_container lrnhist; 
  
  
  // *** Initialize prototype update worker
  // W is passed to this by reference, and is updated internally at each call 
  update_batch_prototypes_prlwkr wkr(X, minX, maxX, W, lambda);
  
  // Set this worker to use mahalanobis distance, if requested 
  if(dist == "Mahalanobis" && MHLdiag.isNotNull()) {
    Rcpp::NumericVector MHLdiag_ = Rcpp::as<Rcpp::NumericVector>(MHLdiag);
    arma::vec MHLdiag__ = Rcpp::as<arma::vec>(MHLdiag_); 
    arma::rowvec MHLdiag___ = MHLdiag__.t(); 
    wkr.set_Mahalanobis_diag(MHLdiag___); 
  }
  
  // *** Initialize the monitoring worker
  VQRecall_worker VQR(W.n_rows, BMUs, QEs);
  // Check for labels, assign them to the monitor if supplied 
  bool is_labeled = false;
  if(XL.isNotNull()) {
    std::vector<std::string> XL_ = Rcpp::as<std::vector<std::string>>(XL);
    VQR.set_XLabel(XL_); 
    is_labeled = true;
  }
  
  
  
  // *** Learning Loop over epochs 
  auto start_time = std::chrono::high_resolution_clock::now(); 
  while(!exit_flag) {
    
    // Increment age
    age++;
    
    // Get new lambda 
    lambda = annealer.current_lambda(age); 
    
    // Update prototypes
    if(parallel) wkr.calc_parallel(); else wkr.calc_serial(); 
    wkr.update_W();
    
    // Compute monitoring measures and their changes 
    BMUs = wkr.BMUs; QEs = wkr.QEs; Costs = wkr.Costs; 
    deltas.update(BMUs, QEs, Costs);
    if(parallel) VQR.calc_parallel(); else VQR.calc_serial(); 
    
    // Store 
    lrnhist.epoch.push_back(age);
    lrnhist.lambda.push_back(lambda);
    lrnhist.cost.push_back(deltas.Cost);
    lrnhist.MQE.push_back(deltas.MQE);
    lrnhist.nhb_effect.push_back(deltas.NhbEffect);
    lrnhist.delCost.push_back(deltas.delCost);
    lrnhist.delMQE.push_back(deltas.delMQE);
    lrnhist.delBMU.push_back(deltas.delBMU);
    lrnhist.Entropy.push_back(VQR.Entropy); 
    if(is_labeled) {
      lrnhist.PurityWOA.push_back(VQR.PurityWOA); 
      lrnhist.WLNumUnique.push_back(VQR.WLTable.size()); 
      lrnhist.WLHellinger.push_back(VQR.WLHellinger); 
    }
    
    // Print, if requested 
    if(verbose) {
      NG_verbose_printer(lrnhist); 
    } 
    
    
    // *** Clear the intermediary results in the parallel worker for the next round 
    wkr.clear(); 
    VQR.clear();
    
    
    // *** Check for convergence
    if(age>1 && lrnhist.delBMU.back() < tol_delBMU) npass_delBMU++; else npass_delBMU = 0;
    if(age>1 && lrnhist.delMQE.back() < tol_delMQE) npass_delMQE++; else npass_delMQE = 0;
    
    if(max_epochs > 0 && int(age) >= max_epochs) {exit_flag = true; break;} 
    if(npass_delBMU >= 3 && npass_delMQE >= 3) {exit_flag = true; break;}
    
    
    // *** Check for user interrupt 
    // Return partial results
    try {Rcpp::checkUserInterrupt();} catch(Rcpp::internal::InterruptedException e) {break;}
  }
  
  // Stop the clock 
  auto stop_time = std::chrono::high_resolution_clock::now(); 
  std::chrono::duration<double, std::ratio<60,1>> exec_time = stop_time - start_time;
  
  // Scale W back to data range 
  W = W*(maxX - minX) + minX; 
  
  // Build output list 
  Rcpp::List out; 
  out["W"] = W; 
  out["epochs"] = age; 
  out["lambda_start"] = lambda0;
  out["lambda_end"] = lambda; 
  out["lambda_decay"] = lambda_decay; 
  out["lambda_schedule"] = lambda_schedule; 
  out["tol_delBMU"] = tol_delBMU; 
  out["tol_delMQE"] = tol_delMQE; 
  out["max_epochs"] = max_epochs; 
  out["exec_time"] = exec_time.count(); 
  out["converged"] = exit_flag; 
  out["LearnHist"] = lrnhist.get_LearnHistDF(); 
  
  return out; 
  
}



