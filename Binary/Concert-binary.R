library(Rcpp)
library(here)

sourceCpp(here("Binary","Concert-binary.cpp"))
#' Load the Concert() function from Rcpp
#'
#' @description
#' This function wraps the compiled C++ implementation of the CONCERT algorithm
#' through Rcpp and provides an R interface.
#'
#' @param X Predictor matrix combining the target and all sources by rows.
#' @param Y Response vector combining the target and all sources.
#' @param n_vec A list specifying sample sizes for the target and each source.
#' @param tau Slab standard deviation in the conditional spike-and-slab prior for \code{beta_k}.
#' @param q_k Prior transferable probability for each source.
#' @param eta Slab standard deviation in the spike-and-slab prior for \code{beta_0}.
#' @param q_0 Prior inclusion probability for the target.
#' @param threshold Convergence threshold for CAVI.
#' @param max_iter Maximum number of iterations for CAVI.
#'
#' @return A list containing the following elements:
#' \itemize{
#'   \item \code{m_beta0}: Estimated slab mean for \code{beta_0}.
#'   \item \code{gamma_Z}: Estimated variational posterior inclusion probability for the target.
#'   \item \code{m_betak}: Estimated slab mean for \code{beta_k}.
#'   \item \code{gamma_I}: Estimated variational posterior transferable probability for each source.
#'   \item \code{V_beta0}: Estimated slab variance for \code{beta_0}.
#'   \item \code{V_betak}: Estimated slab variance for \code{beta_k}.
#' }


sample_ind <- function(n_vec, num_K){
  # The indices of different source data sets
  ind <- list()
  for (k in 1:num_K){
    if (k == 1){
      ind[[1]] <- 1:n_vec[1]
    } else{
      ind[[k]] <- (sum(n_vec[1:(k-1)])+1): sum(n_vec[1:k])
    }
  }
  return(ind)
}

eval_pre <- function(est, X_test=NULL, y_test=NULL){
  pred_err <- NA
  if(!is.null(X_test)& !is.null(y_test)){
    y_pred <- 1/(1+exp(-X_test%*%est)) >= 0.5
    pred_err <- sum(y_test!=y_pred)/length(y_test)
  }
  return(pred_err=pred_err)
}

perf <- function(theta, theta0){
  Itheta0 <- as.numeric(theta0 != 0)
  Itheta <- as.numeric(theta != 0)
  
  TP <- sum(Itheta[which(Itheta0 == 1)])/sum(Itheta0)
  FP <- sum(Itheta[which(Itheta0 == 0)])/max(1,sum(Itheta))
  
  return(c(TP, FP))
}

lasso_est_R <- function(X, y, lambda){
  beta <- as.numeric(glmnet::glmnet(X, y, family="binomial", lambda=lambda)$beta)
  return(beta)
}


################################# R code #################################
### only target
Solo_R <- function(X, y, threshold=1e-6, max_iter=1000, eta=1, q_0=0.05){
  p <- ncol(X)
  n0 <- nrow(X)
  
  trace_m_beta0 <- matrix(0, nrow=p, ncol=1); trace_m_beta0 <- trace_m_beta0[,-1]
  trace_gamma_Z <- matrix(0, nrow=p, ncol=1); trace_gamma_Z <- trace_gamma_Z[,-1]

  #################### Initialization ####################
  m_beta0 <- as.numeric(glmnet::glmnet(X, y, lambda = 0.1)$beta)
  V_beta0 <- rep(0,p)
  gamma_Z <- as.numeric(m_beta0!=0)
  E_W <-  rep(0.5,n0)
  
  #################### CAVI ####################
  diff_m <- 10
  iter_num <- 0
  while (diff_m > threshold){
    iter_start <- Sys.time()
    old_m <- m_beta0
    
    ### beta0: m_beta0, V_beta0
    ### Z: gamma_Z
    order_beta0 <- order(abs(m_beta0), decreasing=T) # prioritized update scheme
    for (j in order_beta0){
      sum_var <- t(X[,j])%*%diag(E_W)%*%X[,j]+1/eta^2
      sum_mean <- t(y-1/2)%*%X[,j]-t(X[,j])%*%diag(E_W)%*%X[,-j]%*%(gamma_Z[-j]*m_beta0[-j])
      V_beta0[j] <- 1/sum_var
      m_beta0[j] <- sum_mean/sum_var
      
      logit <- m_beta0[j]^2/(2*V_beta0[j])+log(q_0*sqrt(V_beta0[j])/((1-q_0)*eta))
      gamma_Z[j] <- 1/(1+exp(-logit))
    }
    trace_m_beta0 <- cbind(trace_m_beta0, m_beta0)
    trace_gamma_Z <- cbind(trace_gamma_Z, gamma_Z)

    # c_omega_0
    m2_beta0 <- diag(gamma_Z*V_beta0)+
      (gamma_Z%*%t(gamma_Z)+diag(gamma_Z*(1-gamma_Z)))*(m_beta0%*%t(m_beta0))
    c_omega_0 <- sqrt(diag(X%*%m2_beta0%*%t(X)))
    E_W <- 1/(2*c_omega_0)*tanh(c_omega_0/2)
    
    diff_m <- sum((m_beta0-old_m)^2)
    iter_end <- Sys.time()
    iter_time <- difftime(iter_end, iter_start)[[1]]
    
    iter_num <- iter_num+1
    # cat("Iter ", iter_num, ": ", diff_m, " with time ", iter_time, " secs.", "\n", sep="")
    if (iter_num > max_iter){
      break
    }
  }
  
  result <- list(m_beta0, gamma_Z, V_beta0)
  names(result) <- c("m_beta0", "gamma_Z", "V_beta0")
  return(result)
}