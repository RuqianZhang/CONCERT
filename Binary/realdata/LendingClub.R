require(here)

# ##################################### data cleaning #####################################
# load(here("Binary","realdata","LC07to11.rda"))
# default_status <- c("Charged Off","Default","Does not meet the credit policy. Status:Charged Off","Late (31-120 days)")
# paid_status <- c("Does not meet the credit policy. Status:Fully Paid","Fully Paid")
# data_subset <- data_subset[data_subset$loan_status %in% c(default_status,paid_status),]
# data_subset$loan_status <- as.numeric(data_subset$loan_status %in% default_status)
# 
# # remove repetitive variables and variables with most values as NA or 0
# data_subset <- data_subset[,c(3,6:8,10,12:17,21,25,26,28:30,33:37,110)]
# data_subset$emp_length[which(data_subset$emp_length=="")] <- 0
# data_subset <- na.omit(data_subset)
# data_subset$term <- as.numeric(data_subset$term=="60 months")
# letter_to_number <- c("A"=7, "B"=6, "C"=5, "D"=4, "E"=3, "F"=2, "G"=1)
# data_subset$grade <- letter_to_number[substr(data_subset$sub_grade, 1, 1)]*10+
#   readr::parse_number(data_subset$sub_grade)*2-1
# data_subset <- data_subset[,-5]
# data_subset$emp_length[which(data_subset$emp_length=="< 1 year")] <- 0.5
# data_subset$emp_length <- readr::parse_number(data_subset$emp_length)
# data_subset$fico <- (data_subset$fico_range_high+data_subset$fico_range_low)/2
# data_subset <- data_subset[-which(data_subset$home_ownership=="OTHER"),]
# data_subset <- fastDummies::dummy_cols(data_subset, select_columns = "home_ownership",
#                                      remove_first_dummy = T, remove_selected_columns = T)
# data_subset$verification_status <- as.numeric(data_subset$verification_status!="Not Verified")
# data_subset <- data_subset[,-c(13,14)] # used to construct fico
# data_subset <- data_subset[,c(10,9,1:7,11:23)] # remove date
# # add pairwise interaction terms
# data_final <- model.matrix(~.^2-1, data=data_subset[,-c(1,2)])
# data_final <- data_final[,-ncol(data_final)] # 0
# data_final <- cbind(data_subset[,c(1,2)], data_final)
# data_final <- data.frame(data_final)
# 
# scale_no_index <- which(apply(data_final, 2, function(x){length(unique(x)) %in% c(1,2)}))
# purposes <- unique(data_final$purpose)
# for (purpose in purposes){
#   data_final[which(data_final$purpose==purpose), -c(1,scale_no_index)] <-
#     apply(data_final[which(data_final$purpose==purpose), -c(1,scale_no_index)], 2, scale)
# }
# # remove NaN
# nan_columns <- sapply(data_final, function(x) any(is.nan(x)))
# data_final <- data_final[,-which(nan_columns==1)]
# save(data_final, file=here("Binary","realdata","LCdata07to11.rda"))


##################################### load data #####################################
load(here("Binary","realdata","LCdata07to11.rda"))
purposes <- unique(data_final$purpose)
discard <- c("car", "home_improvement", "small_business","major_purchase",
             "renewable_energy", "other", "debt_consolidation", "credit_card")
purposes <- setdiff(purposes, discard)

purposes_n <- rep(0,length(purposes))
for (i in 1:length(purposes)){
  purposes_n[i] <- sum(data_final$purpose==purposes[i])
}

data_final <- data_final[which(data_final$purpose %in% purposes),]
table(data_final$purpose,data_final$loan_status)

cor_mat <- cor(data_final[,-c(1,2)])
high_corr <- which(abs(cor_mat) > 0.99, arr.ind = TRUE)
high_corr <- high_corr[high_corr[, 1] != high_corr[, 2], ]
columns_to_remove <- unique(high_corr[, 2])
columns_to_remove <- columns_to_remove[-which(columns_to_remove %in% 1:20)]
data_final <- data_final[,-(columns_to_remove+2)]


##################################### load methods #####################################
require(glmnet)
require(ncvreg)
require(foreach)
require(doParallel)
require(here)

source(here("Binary", "Concert-binary.R"))
require(glmtrans)
require(sparsevb)

##########
target_names <- c("educational", "medical", "wedding", "moving", "house", "vacation")
target_name <- c("house")

for(target_name in target_names){

target_id <- which(data_final$purpose==target_name)
n_tar <- length(target_id)
purposes_source <- setdiff(purposes, target_name)

res <- c()
res_all <- list()
numCores <- 5
registerDoParallel(numCores)

start_seed <- 111
seed_num <- 77
if (target_name == "educational"){
  seed_num <- 111; start_seed <- 111-seed_num*38
} else if (target_name == "medical"){
  seed_num <- 222; start_seed <- 222+seed_num*2
} else if (target_name == "house"){
  start_seed <- seed_num*63
} else if (target_name == "wedding"){
  start_seed <- seed_num*81
} else if (target_name == "moving"){
  start_seed <- 111-seed_num*55
} else if (target_name == "vacation"){
  start_seed <- 111+seed_num*36
}

for (kk in 1:10){
  cat(target_name, kk, "\n")
  results.kk <- foreach(i=((kk-1)*numCores+1):(kk*numCores), .packages = c("sparsevb", "glmnet", "glmtrans")) %dopar%{
    set.seed(start_seed+seed_num*(i-1))
    ########################## train & test ########################## 
    train_id <- sample(target_id, size=0.8*n_tar)
    test_id <- setdiff(target_id, train_id)
    
    data_all <- data_final[,-1]
    
    cons_id_train <- which(apply(data_all[train_id,2:ncol(data_all)],2,sd)==0)
    if (length(cons_id_train)==0){
      data_new <- data_all
    }else{
      data_new <- data_all[,-cons_id_train]
    }
    
    data_train <- as.matrix(data_new[train_id,])
    for (i in 1:length(purposes_source)){
      data_train <- rbind(data_train, as.matrix(data_new[which(data_final$purpose==purposes_source[i]),]))
    }
    data_test <- as.matrix(data_new[test_id,])
    
    X <- data_train[,-1]
    y <- data_train[,1]
    X_test <- data_test[,-1]
    y_test <- data_test[,1]
    
    n_vec <- c(length(train_id), purposes_n[-which(purposes==target_name)]) 
    K <- length(n_vec)-1; n0 <- n_vec[1]; p <- ncol(X)
    
    ########################################################################
    tau=rep(1,K); eta=1; q_0=0.1
    start <- Sys.time()
    threshold <- 0.05
    start <- Sys.time()
    ## tuning based on data
    ridge_0 <- cv.glmnet(X[1:n0,], y[1:n0], family="binomial", alpha=0)
    ridge_beta_0 <- as.numeric(ridge_0$glmnet.fit$beta[,which(ridge_0$lambda==ridge_0$lambda.min)])
    q_k <- 1
    for (k in 2:(K+1)){
      ridge_k <- cv.glmnet(X[sample_ind(n_vec, k)[[k]],], y[sample_ind(n_vec, k)[[k]]], family="binomial", alpha=0)
      ridge_beta_k <- as.numeric(ridge_k$glmnet.fit$beta[,which(ridge_k$lambda==ridge_k$lambda.min)])
      q_k <- min(q_k, sum(abs(ridge_beta_0-ridge_beta_k) <= threshold)/p)
    }
    
    Concert_res <- Concert(X, y, n_vec, tau=tau, q_k=q_k, eta=eta, q_0=q_0)
    end <- Sys.time()
    diff_time <- difftime(end, start, units="secs")[[1]]
    m_beta0 <- Concert_res[[1]]
    gamma_Z <- Concert_res[[2]]
    m_beta0 <- m_beta0*gamma_Z
    prmse_Concert <- unlist(eval_pre(m_beta0, X_test, y_test))

    ###########Sparse VB#############
    Target_res_l <- svb.fit(X[1:n0,], y[1:n0], family="logistic", slab="laplace")
    beta0_target_l <- Target_res_l$mu
    beta0_target_l <- beta0_target_l*Target_res_l$gamma
    prmse_target_l <- unlist(eval_pre(beta0_target_l, X_test, y_test))

    Target_res <- Solo_R(X[1:n0,], y[1:n0], eta=eta, q_0=q_0)
    beta0_target <- Target_res$m_beta0
    beta0_target <- beta0_target*Target_res$gamma_Z
    prmse_target <- unlist(eval_pre(beta0_target, X_test, y_test))

    ###########Lasso#############
    res.lasso <- cv.glmnet(X[1:n0,], y[1:n0], family="binomial")
    beta0_lasso <- as.numeric(res.lasso$glmnet.fit$beta[,which(res.lasso$lambda==res.lasso$lambda.min)])
    prmse_lasso <- unlist(eval_pre(beta0_lasso, X_test, y_test))

    ###########GLMtrans#############
    target <- list(x=X[1:n0,], y=y[1:n0])
    source <- list()
    for (k in 2:(K+1)){
      source_k <- list(x=X[sample_ind(n_vec, k)[[k]],], y=y[sample_ind(n_vec, k)[[k]]])
      source[[k-1]] <- source_k
    }
    glmtrans_fit <- glmtrans(target, source, family="binomial", intercept = F, detection.info = F)
    beta_glmtrans <- glmtrans_fit$beta[-1]
    prmse_glmtrans <- unlist(eval_pre(beta_glmtrans, X_test, y_test))

    list(c(PMSE_Concert=prmse_Concert,
         PMSE_target_l=prmse_target_l,
         PMSE_target = prmse_target,
         PMSE_lasso=prmse_lasso,
         PMSE_glmtrans=prmse_glmtrans),
         Concert_res)
  }
  
  kkkk <- 1
  for (kkk in ((kk-1)*numCores+1):(kk*numCores)){
    res <- c(res,results.kk[[kkkk]][[1]])
    res_all[[kkk]] <- results.kk[[kkkk]][[2]]
    kkkk <- kkkk+1
  }
  measure_mat <- matrix(unlist(res),ncol=5,byrow = TRUE)
  colnames(measure_mat) <- c("Concert","SparseVB","NaiveVB","Lasso","TransGLM")
  ntrial = nrow(measure_mat)
  measure_mat <- rbind(measure_mat,apply(measure_mat[1:ntrial,], 2, mean))
  measure_mat <- rbind(measure_mat,apply(measure_mat[1:ntrial,], 2, sd))
  print(measure_mat[(ntrial+1),])
  
  
  OUTPUT_DIR <- here("Binary","realdata","res")
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  }
  loc <- paste(c(OUTPUT_DIR, "/", target_name,".rda"), collapse ="")
  save_name <- paste(c(target_name, " seed ", seed_num), collapse = "")
  save_res <- list(save_name, measure_mat)
  save(save_res, file=loc)
}
stopImplicitCluster()
}


#####
for(target_name in target_names){
  print(i)
  tissue.tar <- Tissue.tar[i]
  OUTPUT_DIR <- here("Binary","realdata","res")
  loc <- paste(c(OUTPUT_DIR, "/", target_name,".rda"), collapse ="")
  load(loc)
  res_mat <- save_res[[2]]
  ntrial <- nrow(res_mat)-2
  print(round(apply(res_mat[1:ntrial,], 2, mean),3))
  print(round(apply(res_mat[1:ntrial,], 2, sd),3))
}






