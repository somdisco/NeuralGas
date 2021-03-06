# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

.cpp_NGLearn_batch <- function(X, W, tol_delBMU = 0.0, tol_delMQE = 0.0, max_epochs = 999999L, lambda0 = 0.0, lambda_decay = 0.0, lambda_schedule = NULL, XL = NULL, parallel = TRUE, verbose = TRUE, dist = "L2", MHLdiag = NULL) {
    .Call(`_NeuralGas_cpp_NGLearn_batch`, X, W, tol_delBMU, tol_delMQE, max_epochs, lambda0, lambda_decay, lambda_schedule, XL, parallel, verbose, dist, MHLdiag)
}

.cpp_NGLearn_online <- function(X, W, tol_delBMU = 0.0, tol_delMQE = 0.0, max_epochs = 999999L, alpha0 = 0.0, alpha_decay = 0.0, alpha_schedule = NULL, lambda0 = 0.0, lambda_decay = 0.0, lambda_schedule = NULL, XL = NULL, parallel = TRUE, verbose = TRUE) {
    .Call(`_NeuralGas_cpp_NGLearn_online`, X, W, tol_delBMU, tol_delMQE, max_epochs, alpha0, alpha_decay, alpha_schedule, lambda0, lambda_decay, lambda_schedule, XL, parallel, verbose)
}

