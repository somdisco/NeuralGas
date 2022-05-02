# NeuralGas

NeuralGas is an R package for [Neural Gas prototype learning] (https://en.wikipedia.org/wiki/Neural_gas) The main contributions of NeuralGas are: 

+ Fast and efficient C++ implementation of both Batch and Online training (based on [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html) and [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo/index.html))
+ Optional parallel training and recall (as applicable, via [RcppParallel](https://cran.r-project.org/web/packages/RcppParallel/index.html))

# Installation

```r
devtools::install_github("somdisco/NeuralGas")
```

# Documentation

See `docs/NeuralGas-vignette.pdf`for more information.
