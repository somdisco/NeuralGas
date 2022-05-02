.tableau20 = c("blue" = "#1F77B4", 
              "lightblue" = "#73C2FB", 
              "orange" = "#ff7f0e", 
              "lightorange" = "#ffd27f", 
              "green" = "#2ca02c", 
              "lightgreen" = "#98fb98", 
              "red" = "#d62728", 
              "lightred" = "#ffb09c", 
              "purple" = "#b660cd", 
              "lightpurple" = "#e4a0f7", 
              "yellow" = "#ffdb58", 
              "lightyellow" = "#fdfd96", 
              "teal" = "#17becf", 
              "lightteal" = "#c8ffff", 
              "gray" = "#88807b", 
              "lightgray" = "#c7c6c1", 
              "brown" = "#8c564b", 
              "lightbrown"= "#ceb180", 
              "pink" = "#ff6fff", 
              "lightpink"  = "#fde6fa")


#' Visualization of NeuralGas Learning Histories 
#' 
#' @param NGLearnHist a \code{LearnHist} data frame, as returned from a call to \code{\link{NGOnline}} or \code{\link{NGBatch}}.
#' 
#' @details This function is a wrapper to view plots of \code{del_BMU}, \code{del_MQE}, \code{MQE} and Neural Gas's \code{Cost} 
#' as a function of training epoch. 
#' 
#' 
#' @return none, a ggplot is produced
vis_NGLearnHist = function(NGLearnHist) {
  
  ## Convergence   
  
  # Strip out delQE & delBMU, convert to long format 
  plotdf = dplyr::select(NGLearnHist, Epoch, delMQE, delBMU)
  plotdf = tidyr::gather(plotdf, measure, value, delMQE:delBMU) 
  
  # Set plotting colors 
  plotcolors = .tableau20[c('orange','blue')]; names(plotcolors) = c('delMQE','delBMU')
  
  # Build plot 
  ggp_conv = ggplot2::ggplot(plotdf) + 
    ggplot2::geom_line(ggplot2::aes(x=Epoch, y=value/100, group=measure, color=measure), size=1) + 
    ggplot2::scale_color_manual(values = plotcolors) + 
    ggplot2::theme_minimal() + 
    ggplot2::theme(legend.position = c(0.9, 0.9), legend.title = ggplot2::element_blank()) + 
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1)) + 
    ggplot2::xlab('Epoch') + 
    ggplot2::ylab(NULL) + 
    ggplot2::ggtitle('Convergence Measures')
  
  
  ## Costs & MQE 
  plotdf = dplyr::select(NGLearnHist, Epoch, MQE, Cost)
  plotdf = tidyr::gather(plotdf, measure, value, MQE:Cost) 
  plotcolors = .tableau20[c('purple','red')]; names(plotcolors) = c('MQE','Cost')
  
  ggp_qe = ggplot2::ggplot(plotdf) + 
    ggplot2::geom_line(ggplot2::aes(x=Epoch, y=value, group=measure, color=measure), size=1) + 
    ggplot2::scale_color_manual(values = plotcolors) + 
    ggplot2::theme_minimal() + 
    ggplot2::theme(legend.position = c(0.9, 0.9), legend.title = ggplot2::element_blank()) + 
    ggplot2::scale_y_continuous(limits = c(0,max(subset(plotdf, Epoch>1)$value))) + 
    ggplot2::xlab('Epoch') + 
    ggplot2::ylab(NULL) + 
    ggplot2::ggtitle('Quantization Error')
  
  ggpubr::ggarrange(ggp_conv, ggp_qe, nrow = 2)
  
}
