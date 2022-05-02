#ifndef NEURALGAS_HPP
#include "neuralgas.hpp"
#endif


struct dist_prototypes_prlwkr : RcppParallel::Worker {

  // Initial inputs
  const arma::mat& W;
  const arma::rowvec &x;

  // Internval Vars
  unsigned int nW;

  // Outputs
  arma::vec dist; // distance from a single x to all W


  // Constructor
  dist_prototypes_prlwkr(const arma::mat& W, const arma::rowvec& x) : W(W), x(x) {
    nW = W.n_rows;
    dist.set_size(nW); dist.zeros();
  }

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      dist(i) = std::sqrt(arma::accu(arma::square(x-W.row(i))));
    }
  }

  // Parallel invoker
  void calc_parallel() {
    RcppParallel::parallelFor(0, W.n_rows, *this);
  }

  // Parallel invoker
  void calc_serial() {
    for (unsigned int i = 0; i < W.n_rows; i++) {
      dist(i) = std::sqrt(arma::accu(arma::square(x-W.row(i))));
    }
  }
};

struct update_prototypes_prlwkr : RcppParallel::Worker {

  arma::mat& W;
  const double& alpha;
  //const double& lambda;
  const arma::rowvec& x;
  const arma::vec& nhb_factor;
  //const arma::uvec& rank;

  unsigned int nW;

  // Constructor
  // update_prototypes_prlwkr(arma::mat& W, const double& alpha, const double& lambda, const arma::rowvec& x, const arma::uvec& rank) :
  //   W(W), alpha(alpha), lambda(lambda), x(x), rank(rank) {
  //   nW = W.n_rows;
  // }
  update_prototypes_prlwkr(arma::mat& W, const double& alpha, const arma::rowvec& x, const arma::vec& nhb_factor) :
    W(W), alpha(alpha), x(x), nhb_factor(nhb_factor) {
    nW = W.n_rows;
  }



  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++) {

      //double nhb_factor = std::exp(-double(rank(i)) / lambda);

      arma::rowvec diff = x - W.row(i);
      //W.row(i) += alpha * nhb_factor * diff;
      W.row(i) += alpha * nhb_factor(i) * diff;
    }
  }

  // Parallel invoker
  void calc_parallel() {
    RcppParallel::parallelFor(0, W.n_rows, *this);
  }

  // Parallel invoker
  void calc_serial() {

    for (unsigned int i = 0; i < W.n_rows; i++) {

      //double nhb_factor = std::exp(- double(rank(i)) / lambda);

      arma::rowvec diff = x - W.row(i);
      //W.row(i) += alpha * nhb_factor * diff;
      W.row(i) += alpha * nhb_factor(i) * diff;
    }
  }
};

struct update_prototypes_ADAM_prlwkr : RcppParallel::Worker {

  arma::mat& W;
  const double& lambda;
  const arma::rowvec& x;
  const arma::uvec& rank;
  const unsigned int& iter;
  double alpha, beta1, beta2;

  arma::mat m, v;


  unsigned int nW;

  // Constructor
  update_prototypes_ADAM_prlwkr(arma::mat& W, const double& lambda, const arma::rowvec& x, const arma::uvec& rank, const unsigned int& iter,
                                double alpha, double beta1, double beta2) :
    W(W), lambda(lambda), x(x), rank(rank), iter(iter), alpha(alpha), beta1(beta1), beta2(beta2) {
    nW = W.n_rows;
    m = 0*W;
    v = 0*W;
  }


  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++) {

      double nhb_factor = std::exp(-double(rank(i)) / lambda);
      arma::rowvec grad = nhb_factor * (x - W.row(i));

      m.row(i) = beta1*m.row(i) + (1-beta1)*grad;
      v.row(i) = beta2*v.row(i) + (1-beta2)*arma::square(grad);
    }
  }

  // Parallel invoker
  void calc_parallel() {
    RcppParallel::parallelFor(0, W.n_rows, *this);
    double alphat = alpha * std::sqrt(1 - std::pow(beta2,iter)) / (1 - std::pow(beta1,iter));
    W += alphat * m / (arma::sqrt(v) + std::numeric_limits<double>::min());
  }

  // // Parallel invoker
  // void calc_serial() {
  //
  //   for (unsigned int i = 0; i < W.n_rows; i++) {
  //
  //     double nhb_factor = std::exp(- double(rank(i)) / lambda);
  //
  //     arma::rowvec diff = x - W.row(i);
  //     W.row(i) += eps * nhb_factor * diff;
  //   }
  // }
};



// [[Rcpp::export(".cpp_NGLearn_online")]]
Rcpp::List cpp_NGLearn_online(const arma::mat& X, arma::mat W,
                             double tol_delBMU = 0.0, double tol_delMQE = 0.0, int max_epochs = 999999,
                             double alpha0 = 0.0, double alpha_decay = 0.0, Rcpp::Nullable<Rcpp::NumericVector> alpha_schedule = R_NilValue,
                             double lambda0 = 0.0, double lambda_decay = 0.0, Rcpp::Nullable<Rcpp::NumericVector> lambda_schedule = R_NilValue,
                             Rcpp::Nullable<std::vector<std::string>> XL = R_NilValue,
                             bool parallel = true, bool verbose = true) {

  // Inputs:
  // X = data matrix, in external network range
  // minX,maxX = limits of external network range; now compute this internally
  // W = prototype matrix, in internal network range [0,1]
  // lambda = starting lambda value for training
  // decay = decay rate, annealing done at every epoch according to lambda = lambda * decay

  // Setup random generator for sampling indices
  //std::vector<unsigned int> samporder;
  //std::default_random_engine generator;
  //std::uniform_int_distribution<unsigned int> distribution(0,X.n_rows-1);

  // *** Initialize containers related to learning
  double minX = X.min();        // external network range
  double maxX = X.max();
  double alpha = alpha0;        // initial learning rate
  double lambda = lambda0;      // initial neighborhood
  arma::rowvec x = X.row(0);    // the scaled sample @ each iter
  arma::uvec rank(W.n_rows);    // the prototype rank @ each iter
  arma::vec nhb_factor(W.n_rows);
  unsigned int iter = 0;        // num of learning iters performed so far
  unsigned int age = 0;         // num of learning epochs performed so far (epoch = nX iters, nX = nrow(X))
  
  // *** Scale W to [0,1] using min/max of X 
  W = (W - minX) / (maxX - minX); 

  // *** Initialize learning rate annealer
  NG_learnrate_annealer annealer;
  if(alpha_schedule.isNotNull() && lambda_schedule.isNotNull()) {
    Rcpp::NumericVector alpha_schedule_ = Rcpp::as<Rcpp::NumericVector>(alpha_schedule);
    Rcpp::NumericVector lambda_schedule_ = Rcpp::as<Rcpp::NumericVector>(lambda_schedule);
    annealer.set_scheduled_annealing(alpha_schedule_, lambda_schedule_);
  } else if(alpha > 0.0 && alpha_decay > 0.0 && lambda > 0.0 && lambda_decay > 0.0) {
    annealer.set_multiplicative_annealing(alpha, alpha_decay, lambda, lambda_decay);
  } else {
    std::string msg = "Cannot parse lambda annealing. Input either:\n";
    msg += "(1) positive values for alpha, alpha_decay, lambda, lambda_decay, or\n";
    msg += "(2) an alpha_schedule and lambda_schedule";
    Rcpp::stop(msg);
  }


  // *** Initialize containers related to monitoring & reporting
  unsigned int npass_delBMU = 0;  // num of consecutive monitorings the delBMU tol has been met
  unsigned int npass_delMQE = 0;   // num of consecutive monitorings the delQE tol has been met
  bool exit_flag = false;         // whether all conv criteria have been met


  arma::uvec nXsampled(X.n_rows);  // num of times each data vector has been sampled
  nXsampled.zeros();
  arma::uvec BMUs(X.n_rows);
  BMUs.fill(std::numeric_limits<unsigned int>::quiet_NaN());
  //BMUs.fill();
  arma::vec QEs(X.n_rows);         // QE of all data @ current monitoring
  QEs.fill(arma::datum::nan);
  arma::vec Costs(X.n_rows);         // QE of all data @ current monitoring
  Costs.fill(arma::datum::nan);

  NG_deltas deltas;
  deltas.update(BMUs, QEs, Costs);

  // Learn history storage
  NG_learnhist_container lrnhist;

  // *** Initialize prototype distance & update workers
  dist_prototypes_prlwkr DistWorker(W, x);
  update_prototypes_prlwkr UpdateWorker(W, alpha, x, nhb_factor);


  // *** Initialize the monitoring worker
  // If labels were given, assign them to the worker as well
  VQRecall_worker VQR(W.n_rows, BMUs, QEs);
  bool is_labeled = false;
  if(XL.isNotNull()) {
    std::vector<std::string> XL_ = Rcpp::as<std::vector<std::string>>(XL);
    VQR.set_XLabel(XL_); 
    is_labeled = true;  
  }



  // *** Start learning
  auto start_time = std::chrono::high_resolution_clock::now();
  while(!exit_flag) {

    iter++;

    // Get new lambda
    alpha = annealer.current_alpha(iter);
    lambda = annealer.current_lambda(iter);


    // Sample an x and scale it to [0,1]
    //unsigned int Xidx = distribution(generator);
    Rcpp::IntegerVector thissample = Rcpp::sample(X.n_rows, 1, true, R_NilValue, false); // (n, size, replace, probs, 1-based)
    unsigned int Xidx = (unsigned int)thissample(0);
    //samporder.push_back(Xidx);
    nXsampled(Xidx)++;
    x = (X.row(Xidx) - minX) / (maxX - minX);

    // Compute distances to all prototypes, rank them
    DistWorker.calc_parallel();
    rank_prototypes(rank, BMUs(Xidx), QEs(Xidx), DistWorker.dist);

    // Compute neighborhood factor from this x to all W
    nhb_factor = arma::exp(-arma::conv_to<arma::vec>::from(rank) / lambda);

    // Compute contribution to cost from this x
    Costs(Xidx) = arma::accu(nhb_factor % DistWorker.dist);

    // Update prototypes
    if(parallel) UpdateWorker.calc_parallel(); else UpdateWorker.calc_serial();


    if( iter % X.n_rows == 0) {
      // Increment age
      age++;

      // Compute monitoring measures and their changes
      deltas.update(BMUs, QEs, Costs);
      if(parallel) VQR.calc_parallel(); else VQR.calc_serial(); 

      // Store measures
      lrnhist.epoch.push_back(age);
      lrnhist.alpha.push_back(alpha);
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
      

      // Print measures, if requested
      if(verbose) {
        NG_verbose_printer(lrnhist); 
      } 


      // *** Check for convergence
      if(age>1 && lrnhist.delBMU.back() < tol_delBMU) npass_delBMU++; else npass_delBMU = 0;
      if(age>1 && lrnhist.delMQE.back() < tol_delMQE) npass_delMQE++; else npass_delMQE = 0;

      if(max_epochs > 0 && int(age) >= max_epochs) {exit_flag = true; break;}
      if(npass_delBMU >= 3 && npass_delMQE >= 3) {exit_flag = true; break;}


      // *** Check for user interrupt
      // Return partial results
      try {Rcpp::checkUserInterrupt();} catch(Rcpp::internal::InterruptedException e) {break;}

      // *** Clear the intermediary results in the parallel worker for the next round
      VQR.clear();

    } // close epoch monitoring

  } // close learning

  // Stop the clock
  auto stop_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::ratio<60,1>> exec_time = stop_time - start_time;

  // Scale W back to data range
  W = W*(maxX - minX) + minX;

  // Build output list
  Rcpp::List out;
  out["W"] = W;
  out["iterations"] = iter;
  out["epochs"] = age;
  out["alpha_start"] = alpha0;
  out["alpha_end"] = alpha;
  out["alpha_decay"] = alpha_decay;
  out["alpha_schedule"] = alpha_schedule;
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
  //out["samporder"] = samporder;

  return out;

}


