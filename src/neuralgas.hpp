#ifndef NEURALGAS_HPP
#define NEURALGAS_HPP

#ifndef RcppArmadillo_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#endif

#ifndef RcppParallel_H
#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]
#endif

// [[Rcpp::plugins(cpp11)]]

#include <chrono> // for measuring execution time 

#ifndef VQTOOLS_VQTOOLS_HPP
#include "VQTools.hpp"
#endif



// Function to rank the distances from a single vector x to all prototypes 
// Rank is done in ascending order, so that rank(i) = 0 means dist(x,Wi) is SMALLEST value in vector. 
// For speed, the BMU index and associated quantization error is also returned. 
// First 3 arguments are return values. 
inline void rank_prototypes(arma::uvec& rank, unsigned int& BMU, double& QE, const arma::vec& dW) {
  
  // Find the indexes whose order would sort dW.
  // The first of these indices is the BMU, so we can set it and QE now
  arma::uvec sortidx = arma::sort_index(dW, "ascend"); 
  BMU = sortidx(0); 
  QE = dW(BMU); 
  
  // Must sort again to get ranks
  rank = arma::sort_index(sortidx, "ascend"); 
  
  return; 
}

inline void rank_prototypes(arma::uvec& rank, unsigned int& BMU, double& QE, const arma::rowvec& x, const arma::mat& W) {
  
  // Compute distance from x to all W 
  arma::mat Wminusx = W; 
  Wminusx.each_row() -= x; 
  arma::vec dW = arma::sqrt(arma::sum(arma::pow(Wminusx, 2.0),1));
  
  rank_prototypes(rank, BMU, QE, dW); 
  
  // Find the indexes whose order would sort dW.
  // The first of these indices is the BMU, so we can set it and QE now
  // arma::uvec sortidx = arma::sort_index(dW, "ascend"); 
  // BMU = sortidx(0); 
  // QE = dW(BMU); 
  // 
  // // Must sort again to get ranks
  // rank = arma::sort_index(sortidx, "ascend"); 
  
  return; 
}

inline void rank_prototypes(arma::uvec& rank, unsigned int& BMU, double& QE, arma::vec& dW, const arma::rowvec& x, const arma::mat& W) {
  
  // Compute distance from x to all W 
  arma::mat Wminusx = W; 
  Wminusx.each_row() -= x; 
  dW = arma::sqrt(arma::sum(arma::pow(Wminusx, 2.0),1));
  
  rank_prototypes(rank, BMU, QE, dW); 
  
  return; 
}

inline void rank_prototypes_mahalanobis_diag(arma::uvec& rank, unsigned int& BMU, double& QE, arma::vec& dW, const arma::rowvec& x, const arma::mat& W, const arma::rowvec& mahal_diag) {
  
  // Compute distance from x to all W 
  arma::mat Wminusx = W; 
  Wminusx.each_row() -= x; // Computing (W-x)
  Wminusx = arma::pow(Wminusx, 2.0); // Computing (W-x)^2
  Wminusx.each_row() %= mahal_diag; // Computing weight * (W-x)^2
  dW = arma::sqrt(arma::sum(Wminusx,1)); // Computing sqrt(row sums)
  
  rank_prototypes(rank, BMU, QE, dW); 
  
  return; 
}

inline arma::vec dist_to_protos_L2(const arma::rowvec& x, const arma::mat& W) {
  // Compute distance from x to all rows of W 
  arma::mat Wminusx = W; 
  Wminusx.each_row() -= x; // Computing (W-x)
  Wminusx = arma::pow(Wminusx, 2.0); // Computing (W-x)^2
  arma::vec dW = arma::sqrt(arma::sum(Wminusx,1)); // Computing sqrt(row sums)
  return dW; 
}

inline arma::vec dist_to_protos_Mahalanobis_diag(const arma::rowvec& x, const arma::mat& W, const arma::rowvec& Mdiag) {
  // Compute distance from x to all W 
  arma::mat Wminusx = W; 
  Wminusx.each_row() -= x; // Computing (W-x)
  Wminusx = arma::pow(Wminusx, 2.0); // Computing (W-x)^2
  Wminusx.each_row() /= Mdiag; // Computing weight * (W-x)^2
  arma::vec dW = arma::sqrt(arma::sum(Wminusx,1)); // Computing sqrt(row sums)
  
  return dW; 
}

struct VQQuality_worker : public RcppParallel::Worker {
  
  // Inputs 
  const arma::uvec& BMU;
  const unsigned int& nW; 
  
  // Optional Inputs 
  arma::uvec XLabel; 
  
  // Internal variables 
  bool is_labeled; 
  std::map<unsigned int,unsigned int> XLabelMap; 
  
  // outputs 
  arma::uvec RFSize;
  arma::uvec RFLabel; 
  arma::vec RFLabelPurity; 
  
  double RFEntropy; 
  double RFLWPurity, RFLUPurity; // weighted & unweighted average purity scores
  unsigned int RFLNumUnique; // number of unique RFLabels 
  double RFLHellinger; // Hellinger distance between categorical distributions of XLabel & RFLabel 
  

  // Constructor 
  VQQuality_worker(const arma::uvec& BMU, const unsigned int& nW)
    : BMU(BMU), nW(nW)
  {
    // Initialize outputs 
    is_labeled = false; 
    RFSize.set_size(nW); 
    RFSize.zeros(); 
  }
  
  
  // Set labels, if available 
  void set_XLabel(const arma::uvec& XLabel_) {
    if(XLabel_.n_elem != BMU.n_elem) Rcpp::stop("length(XLabel) != nW");
    
    XLabel = XLabel_; 
    is_labeled = true; 
    
    for(unsigned int i=0; i<XLabel.n_elem; ++i) {
      XLabelMap[XLabel(i)]++; 
    }
    
    RFLabel.set_size(nW); 
    RFLabel.fill(std::numeric_limits<unsigned int>::max());
    RFLabelPurity.set_size(nW); RFLabelPurity.zeros(); 
  }
  
  
  // Set the map information for a single receptive field 
  void single_RF(unsigned int i) {
    
    // Find RF members, set RFSize. If =0, exit 
    arma::uvec RFMembers = arma::find(BMU == i); 
    RFSize(i) = RFMembers.n_elem; 
    if(RFSize(i) == 0) return; 
    
    // If no labeled data, return 
    if(!is_labeled) return; 
    
    // Tabulate frequency of labels in this RF 
    std::map<unsigned int, unsigned int> RFLabelMap; 
    for(unsigned int m=0; m<RFSize(i); ++m) {
      RFLabelMap[XLabel(RFMembers(m))]++;
    }
    
    // Get Iterator to largest value in map 
    std::map<unsigned int, unsigned int>::const_iterator maxit = std::max_element(RFLabelMap.cbegin(), RFLabelMap.cend(),
                                                                                  [](const std::pair<unsigned int, unsigned int>& p1, const std::pair<unsigned int, unsigned int>& p2) {
                                                                                    return p1.second < p2.second; });

    RFLabel(i) = maxit->first;
    RFLabelPurity(i) = double(maxit->second) / RFSize(i);
  }
  
  
  void clear() {
    RFSize.zeros(); 
    RFEntropy = 0.0; 
    
    if(is_labeled) {
      RFLabel.fill(std::numeric_limits<unsigned int>::max());
      RFLabelPurity.zeros(); 
      RFLWPurity = 0.0;
      RFLUPurity = 0.0; // weighted & unweighted average purity scores
      RFLNumUnique = 0; // number of unique RFLabels 
      RFLHellinger = 0.0; // Hellinger distance between categorical distributions of XLabel & RFLabel 
    }
  }
  
  // Parallel operator - find BMU of each row of X in parallel
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++) {
      single_RF(i);
    }
  }
  
  // Parallel call method
  void calc_parallel() {
    RcppParallel::parallelFor(0, nW, *this);
    this->calc_global_measures();
  }
  
  // Non-parallel call method
  void calc_serial() {
    // Find BMU of each row of X
    for(unsigned int i=0; i<nW; ++i) {
      single_RF(i);
    }

    this->calc_global_measures();    
  }
  
  void calc_global_measures() {
    
    // Strip out the active RFs (those with nonzero size) 
    arma::uvec activeRF = arma::find(RFSize); 
    unsigned int nactiveRF = activeRF.n_elem; 
    
    // *** Normalized entropy of the quantization
    // Can only do this for active RFs to avoid log(0) in the calculation 
    arma::vec pRF = arma::conv_to<arma::vec>::from(RFSize.elem(activeRF)) / double(arma::accu(RFSize)); 
    RFEntropy = -arma::accu(pRF % arma::log(pRF)) / std::log(nW); 
    
    // If no labeled data, return
    if(!is_labeled) return;


    // *** Un-weighted Average Purity Score
    // Ignore the dead RFs as they are not labeled
    RFLUPurity = arma::mean(RFLabelPurity.elem(activeRF));

    // *** Weighted Average Purity Score
    // Use pRF as weights, since they are already calculated
    RFLWPurity = arma::accu(RFLabelPurity.elem(activeRF) % pRF);

    // *** Count the unique RF labels
    arma::uvec unq_RFLabels = arma::unique(RFLabel.elem(activeRF));
    RFLNumUnique = unq_RFLabels.n_elem;


    // *** Hellinger Distance between the categorical distributions of Xlabel and RFLabel
    // We already have a frequency table of Xlabels, computed during set_XLabel().
    // Turn the frequencies into proportions, and compute the corresponding proportion vector
    // for the distribution of RFLabels
    arma::vec pXLabel(XLabelMap.size()), pRFLabel(XLabelMap.size()); pRFLabel.zeros();
    std::map<unsigned int, unsigned int>::const_iterator Xit = XLabelMap.cbegin();
    for(unsigned int j=0; j<XLabelMap.size(); ++j) {
      // XLabel proportions
      pXLabel(j) = double(Xit->second) / double(BMU.n_elem);

      // RFLabel proportions
      unsigned int this_label = Xit->first; 
      //arma::uvec is_this_label = arma::find(RFLabel.elem(activeRF) == this_label);
      arma::uvec is_this_label = arma::find(RFLabel == this_label);
      pRFLabel(j) = double(is_this_label.n_elem) / double(nactiveRF);

      // Advance the XLabelMap iterator
      Xit++;
    }

    // Compute the Battacharyya coefficient between the categorical distributions
    double BC = arma::accu(arma::sqrt(pXLabel % pRFLabel));
    //Rcpp::Rcout << " " << BC << " " << pXLabel.max() << " " << arma::accu(pXLabel) << " " <<  pRFLabel.max() << " " << arma::accu(pRFLabel) << std::endl; 
    
    // Hellinger distance
    RFLHellinger = std::sqrt(1.0 - BC);
    
    // Finish 
    return; 
  }
};

// struct NG_learnhist_container {
//   std::vector<unsigned int> epoch; 
//   std::vector<double> alpha;
//   std::vector<double> lambda;
//   std::vector<double> cost; 
//   std::vector<double> MQE;
//   std::vector<double> nhb_effect;
//   std::vector<double> delCost; 
//   std::vector<double> delQE; 
//   std::vector<double> delBMU; 
//   std::vector<double> RFEntropy;
//   std::vector<double> RFLWPurity, RFLUPurity;
//   std::vector<double> RFLHellinger; 
//   std::vector<unsigned int> RFLNumUnique;
// 
//   Rcpp::DataFrame get_LearnHistDF() {
//     Rcpp::DataFrame out;
//     
//     if(epoch.size() > 0) out.push_back(epoch, "Epoch"); 
//     if(alpha.size() > 0) out.push_back(alpha, "alpha");
//     if(lambda.size() > 0) out.push_back(lambda, "lambda"); 
//     if(cost.size() > 0) out.push_back(cost, "Cost");
//     if(MQE.size() > 0) out.push_back(MQE, "MQE");
//     if(nhb_effect.size() > 0) out.push_back(nhb_effect, "NhbEff");
//     if(delCost.size() > 0) out.push_back(delCost, "delCost");
//     if(delQE.size() > 0) out.push_back(delQE, "delQE");
//     if(delBMU.size() > 0) out.push_back(delBMU, "delBMU"); 
//     
//     if(RFEntropy.size() > 0) out.push_back(RFEntropy, "RFEnt");
//     
//     if(RFLWPurity.size() > 0) out.push_back(RFLWPurity, "RFLWPur");
//     if(RFLUPurity.size() > 0) out.push_back(RFLUPurity, "RFLUPur");
//     if(RFLNumUnique.size() > 0) out.push_back(RFLNumUnique, "RFLUnq");
//     if(RFLHellinger.size() > 0) out.push_back(RFLHellinger, "RFLHell");
//     
//     out = Rcpp::as<Rcpp::DataFrame>(out);
//     
//     return out; 
//   }
// };

struct NG_learnhist_container {
  std::vector<unsigned int> epoch; 
  std::vector<double> alpha;
  std::vector<double> lambda;
  std::vector<double> cost; 
  std::vector<double> MQE;
  std::vector<double> nhb_effect;
  std::vector<double> delCost; 
  std::vector<double> delMQE; 
  std::vector<double> delBMU; 
  std::vector<double> Entropy;
  std::vector<double> PurityWOA, PurityUOA;
  std::vector<double> WLHellinger; 
  std::vector<unsigned int> WLNumUnique;
  
  Rcpp::DataFrame get_LearnHistDF() {
    Rcpp::DataFrame out;
    
    if(epoch.size() > 0) out.push_back(epoch, "Epoch"); 
    if(alpha.size() > 0) out.push_back(alpha, "alpha");
    if(lambda.size() > 0) out.push_back(lambda, "lambda"); 
    if(cost.size() > 0) out.push_back(cost, "Cost");
    if(MQE.size() > 0) out.push_back(MQE, "MQE");
    if(nhb_effect.size() > 0) out.push_back(nhb_effect, "NhbEff");
    if(delCost.size() > 0) out.push_back(delCost, "delCost");
    if(delMQE.size() > 0) out.push_back(delMQE, "delMQE");
    if(delBMU.size() > 0) out.push_back(delBMU, "delBMU"); 
    
    if(Entropy.size() > 0) out.push_back(Entropy, "Entropy");
    
    if(PurityWOA.size() > 0) out.push_back(PurityWOA, "PurityWOA");
    if(PurityUOA.size() > 0) out.push_back(PurityUOA, "PurityUOA");
    if(WLNumUnique.size() > 0) out.push_back(WLNumUnique, "WLUnq");
    if(WLHellinger.size() > 0) out.push_back(WLHellinger, "WLHell");
    
    out = Rcpp::as<Rcpp::DataFrame>(out);
    
    return out; 
  }
};

struct NG_deltas{
  
  arma::uvec prvBMUs; 
  arma::vec prvQEs; 
  arma::vec prvCosts; 
  
  arma::uvec prvActive; 
  
  double delBMU; 

  //double delQE; 
  
  double MQE; 
  double prvMQE; 
  double delMQE; 
  
  double Cost; 
  double prvCost; 
  double delCost; 
  
  double NhbEffect; 
  double prvNhbEffect; 
  
 
  void update(const arma::uvec& newBMUs, const arma::vec& newQEs, const arma::vec& newCosts) {
    
    // Update active values 
    arma::uvec Active = arma::find_finite(newQEs); 

    // Update absolute quantities 
    prvMQE = MQE; 
    prvCost = Cost; 
    prvNhbEffect = NhbEffect; 
    if(Active.n_elem == 0) {
      MQE = std::numeric_limits<double>::quiet_NaN();
      Cost = std::numeric_limits<double>::quiet_NaN();
      NhbEffect = std::numeric_limits<double>::quiet_NaN();
    } else {
      MQE = arma::mean(newQEs.elem(Active)); 
      Cost = arma::mean(newCosts.elem(Active)); 
      NhbEffect = Cost / MQE; 
    }
    
    
    // Update relative quantities 
    if(prvActive.n_elem == 0) {
      delBMU = std::numeric_limits<double>::quiet_NaN();
      delMQE = std::numeric_limits<double>::quiet_NaN();
      //delQE = std::numeric_limits<double>::quiet_NaN();
      delCost = std::numeric_limits<double>::quiet_NaN();
    } else {
      arma::uvec nBMUmismatch = arma::find(newBMUs.elem(prvActive) != prvBMUs.elem(prvActive));
      delBMU = double(nBMUmismatch.n_elem) / double(Active.n_elem) * 100.0;
      
      delMQE = std::abs(MQE - prvMQE) / (prvMQE + std::numeric_limits<double>::min()) * 100.0; 
      
      //arma::vec delQEs = (prvQEs.elem(prvActive) - newQEs.elem(prvActive)) / (prvQEs.elem(prvActive) + std::numeric_limits<double>::min());
      //arma::uvec prvQE_zero = prvActive.elem(arma::find(!(prvQEs.elem(prvActive) > 0))); 
      //arma::vec delQEs = arma::abs(prvQEs.elem(prvQE_nonzero) - newQEs.elem(prvQE_nonzero)) / (prvQEs.elem(prvQE_nonzero) + std::numeric_limits<double>::min());
      //delQE = arma::mean(arma::abs( delQEs )) * 100.0;  
      //delQE = delMQE * 100.0; 
      
      delCost = std::abs(Cost - prvCost) / prvCost * 100; 
    }
    
    
    
    // Update vector quantities 
    prvBMUs = newBMUs; 
    
    prvQEs = newQEs; 
    
    prvCosts = newCosts; 
    
    prvActive = Active; 
  }
  
  
  
};

struct NG_learnrate_annealer {
  
  // *** Variables 
  
  std::string anneal_type; 
  
  // For multiplicative annealing 
  double alpha_start; 
  double alpha_mult_decay; 
  double lambda_start; 
  double lambda_mult_decay; 
  
  // For scheduled annealing 
  std::map<unsigned int, double> alpha_schedule; 
  std::map<unsigned int, double> lambda_schedule; 
  
  
  // *** Different constructors 
  
  // For Online + multiplicative decay 
  void set_multiplicative_annealing(double alpha_start_, double alpha_mult_decay_, double lambda_start_, double lambda_mult_decay_) {
    anneal_type = "multiplicative";
    alpha_start = alpha_start_; 
    alpha_mult_decay = alpha_mult_decay_; 
    lambda_start = lambda_start_; 
    lambda_mult_decay = lambda_mult_decay_; 
  }
  
  // For Batch + multiplicative decay 
  void set_multiplicative_annealing(double lambda_start_, double lambda_mult_decay_) {
    anneal_type = "multiplicative";
    lambda_start = lambda_start_; 
    lambda_mult_decay = lambda_mult_decay_; 
  }
  
  // For online + scheduled annealing 
  void set_scheduled_annealing(Rcpp::NumericVector alpha_schedule_vec, Rcpp::NumericVector lambda_schedule_vec) {
    anneal_type = "scheduled";
    alpha_schedule = this->decode_schedule(alpha_schedule_vec);
    lambda_schedule = this->decode_schedule(lambda_schedule_vec); 
  }

  // For batch + scheduled annealing 
  void set_scheduled_annealing(Rcpp::NumericVector lambda_schedule_vec) {
    anneal_type = "scheduled";
    lambda_schedule = this->decode_schedule(lambda_schedule_vec); 
  }
  
  
  // *** Function to turn a named numeric vector into a std::map for scheduled annealing
  // (names = schedule epoch changes, values = rate at scheduled epoch changes) 
  std::map<unsigned int, double> decode_schedule(Rcpp::NumericVector schedule) {
    
    Rcpp::CharacterVector epoch_names = schedule.names(); 
    arma::uvec epochs(schedule.size()); 
    std::map<unsigned int, double> schedule_; 
    
    for(unsigned int i=0; i<schedule.size(); ++i) {
      std::string this_epoch = Rcpp::as<std::string>(epoch_names[i]); 
      epochs(i) = std::stoul(this_epoch);
      schedule_[epochs(i)] = schedule(i);
    }
    
    schedule_[std::numeric_limits<unsigned int>::max()] = schedule(epochs.index_max()); 
    
    return schedule_;
  }
  
  
  // *** Functions to retrieve annealed learning rates at a given age 
  // The values returned are the rates that are effective at the BEGINNING of the current learning iteration, given a network age 
  double current_alpha(unsigned int age) {
    
    double alpha = 0.0; 
    
    if(anneal_type == "multiplicative") {
      alpha = alpha_start * std::pow(alpha_mult_decay, age-1);
    } else if(anneal_type == "scheduled") {
      alpha = alpha_schedule.lower_bound(age) ->second; 
    }
    
    return alpha; 
  }
  
  double current_lambda(unsigned int age) {
    
    double lambda = 0.0; 
    
    if(anneal_type == "multiplicative") {
      lambda = lambda_start * std::pow(lambda_mult_decay, age-1);
    } else if(anneal_type == "scheduled") {
      lambda = lambda_schedule.lower_bound(age) ->second; 
    }
    
    return std::max(lambda, 1e-8); 
  }
  
};

inline void NG_verbose_printer(const NG_learnhist_container& lrnhist) {
  
  // Get index of last entry into container's storage vectors, 
  // extract current epoch 
  unsigned int curidx = lrnhist.epoch.size() - 1; 
  unsigned int curepoch = lrnhist.epoch[curidx]; 
  
  // Determine whether learn hist for labels exist 
  bool is_labeled = false; 
  if(lrnhist.PurityWOA.size() > 0) is_labeled = true; 
  
  // Print headers first, and after every 10 reports 
  if(curepoch % 10 == 1) {
    
    Rprintf("%5s", "Epoch"); 
    if(lrnhist.alpha.size() > 0) Rprintf("%10s", "alpha");
    
    Rprintf("%10s", "lambda");
    Rprintf("%10s", "Cost");
    Rprintf("%10s", "MQE");
    Rprintf("%10s", "NhbEff");
    Rprintf("%10s", "delCost");
    Rprintf("%10s", "delMQE");
    Rprintf("%10s", "delBMU");
    Rprintf("%10s", "Entropy");
    
    if(is_labeled) {
      Rprintf("%10s", "PurityWOA");
      Rprintf("%7s", "WLUnq");
      Rprintf("%8s", "WLHell");
    } 
    Rprintf("\n");
  }
  
  Rprintf("%5u", curepoch);
  if(lrnhist.alpha.size() > 0) Rprintf("%10.3f", lrnhist.alpha[curidx]);
  Rprintf("%10.3f", lrnhist.lambda[curidx]);
  Rprintf("%10.5f", lrnhist.cost[curidx]); 
  Rprintf("%10.5f", lrnhist.MQE[curidx]);
  Rprintf("%10.3f", lrnhist.nhb_effect[curidx]);
  Rprintf("%10.3f", lrnhist.delCost[curidx]);
  Rprintf("%10.3f", lrnhist.delMQE[curidx]);
  Rprintf("%10.3f", lrnhist.delBMU[curidx]);
  Rprintf("%10.3f", lrnhist.Entropy[curidx]);
  if(is_labeled) {
    Rprintf("%10.3f", lrnhist.PurityWOA[curidx]); 
    Rprintf("%7d", lrnhist.WLNumUnique[curidx]);
    Rprintf("%8.3f", lrnhist.WLHellinger[curidx]);
  }
  Rprintf("\n");
} 

  


#endif
