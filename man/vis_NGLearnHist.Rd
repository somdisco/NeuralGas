% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/vis_NGLearnHist.R
\name{vis_NGLearnHist}
\alias{vis_NGLearnHist}
\title{Visualization of NeuralGas Learning Histories}
\usage{
vis_NGLearnHist(NGLearnHist)
}
\arguments{
\item{NGLearnHist}{a \code{LearnHist} data frame, as returned from a call to \code{\link{NGOnline}} or \code{\link{NGBatch}}.}
}
\value{
none, a ggplot is produced
}
\description{
Visualization of NeuralGas Learning Histories
}
\details{
This function is a wrapper to view plots of \code{del_BMU}, \code{del_QE}, \code{MQE} and Neural Gas's \code{Cost} 
as a function of training epoch.
}
