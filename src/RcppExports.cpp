// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// cpp_NGLearn_batch
Rcpp::List cpp_NGLearn_batch(const arma::mat& X, arma::mat W, double tol_delBMU, double tol_delMQE, int max_epochs, double lambda0, double lambda_decay, Rcpp::Nullable<Rcpp::NumericVector> lambda_schedule, Rcpp::Nullable<std::vector<std::string>> XL, bool parallel, bool verbose, std::string dist, Rcpp::Nullable<Rcpp::NumericVector> MHLdiag);
RcppExport SEXP _NeuralGas_cpp_NGLearn_batch(SEXP XSEXP, SEXP WSEXP, SEXP tol_delBMUSEXP, SEXP tol_delMQESEXP, SEXP max_epochsSEXP, SEXP lambda0SEXP, SEXP lambda_decaySEXP, SEXP lambda_scheduleSEXP, SEXP XLSEXP, SEXP parallelSEXP, SEXP verboseSEXP, SEXP distSEXP, SEXP MHLdiagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type W(WSEXP);
    Rcpp::traits::input_parameter< double >::type tol_delBMU(tol_delBMUSEXP);
    Rcpp::traits::input_parameter< double >::type tol_delMQE(tol_delMQESEXP);
    Rcpp::traits::input_parameter< int >::type max_epochs(max_epochsSEXP);
    Rcpp::traits::input_parameter< double >::type lambda0(lambda0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda_decay(lambda_decaySEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type lambda_schedule(lambda_scheduleSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::string>> >::type XL(XLSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< std::string >::type dist(distSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type MHLdiag(MHLdiagSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_NGLearn_batch(X, W, tol_delBMU, tol_delMQE, max_epochs, lambda0, lambda_decay, lambda_schedule, XL, parallel, verbose, dist, MHLdiag));
    return rcpp_result_gen;
END_RCPP
}
// cpp_NGLearn_online
Rcpp::List cpp_NGLearn_online(const arma::mat& X, arma::mat W, double tol_delBMU, double tol_delMQE, int max_epochs, double alpha0, double alpha_decay, Rcpp::Nullable<Rcpp::NumericVector> alpha_schedule, double lambda0, double lambda_decay, Rcpp::Nullable<Rcpp::NumericVector> lambda_schedule, Rcpp::Nullable<std::vector<std::string>> XL, bool parallel, bool verbose);
RcppExport SEXP _NeuralGas_cpp_NGLearn_online(SEXP XSEXP, SEXP WSEXP, SEXP tol_delBMUSEXP, SEXP tol_delMQESEXP, SEXP max_epochsSEXP, SEXP alpha0SEXP, SEXP alpha_decaySEXP, SEXP alpha_scheduleSEXP, SEXP lambda0SEXP, SEXP lambda_decaySEXP, SEXP lambda_scheduleSEXP, SEXP XLSEXP, SEXP parallelSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type W(WSEXP);
    Rcpp::traits::input_parameter< double >::type tol_delBMU(tol_delBMUSEXP);
    Rcpp::traits::input_parameter< double >::type tol_delMQE(tol_delMQESEXP);
    Rcpp::traits::input_parameter< int >::type max_epochs(max_epochsSEXP);
    Rcpp::traits::input_parameter< double >::type alpha0(alpha0SEXP);
    Rcpp::traits::input_parameter< double >::type alpha_decay(alpha_decaySEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type alpha_schedule(alpha_scheduleSEXP);
    Rcpp::traits::input_parameter< double >::type lambda0(lambda0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda_decay(lambda_decaySEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type lambda_schedule(lambda_scheduleSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::string>> >::type XL(XLSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_NGLearn_online(X, W, tol_delBMU, tol_delMQE, max_epochs, alpha0, alpha_decay, alpha_schedule, lambda0, lambda_decay, lambda_schedule, XL, parallel, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_NeuralGas_cpp_NGLearn_batch", (DL_FUNC) &_NeuralGas_cpp_NGLearn_batch, 13},
    {"_NeuralGas_cpp_NGLearn_online", (DL_FUNC) &_NeuralGas_cpp_NGLearn_online, 14},
    {NULL, NULL, 0}
};

RcppExport void R_init_NeuralGas(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}