#include <RcppArmadillo.h>
#include <iostream>
#include <stdio.h>
#include <Rcpp/Benchmark/Timer.h>

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
vec sample_ind_C(vec n_vec, int num_K);
vec lasso_est(mat X, vec y, double lambda);

// [[Rcpp::export]]
List Concert(mat X, vec y, vec n_vec,
            vec tau, double q_k, double eta, double q_0,
            double threshold=1e-6, int max_iter=1000){
	int p=X.n_cols, K=n_vec.n_elem-1;
  vec ind=sample_ind_C(n_vec, K+1);
	
	Timer timer;
	timer.step("start");
	
  //*************************************************************
	vec m_beta0, gamma_Z(p,fill::zeros), V_beta0(p,fill::ones);
  mat m_betak(p,K), gamma_I(p,K,fill::zeros), V_betak(p,K,fill::ones);
  vec E_W;
	vec tmp_vec;
  uvec index;
	//*************************************************************
	// Initialization
  m_beta0=lasso_est(X.rows(ind(0),ind(1)), y.subvec(ind(0),ind(1)), 0.01);  
  index=find(m_beta0!=0);
  gamma_Z(index)=linspace(1,1,index.n_elem);
  for (int k=1; k<K+1; ++k){
    m_betak.col(k-1)=lasso_est(X.rows(ind(k)+1,ind(k+1)), y.subvec(ind(k)+1,ind(k+1)), 0.01);
    index=find(m_betak.col(k-1)!=0);
    tmp_vec=linspace(0,0,p); tmp_vec(index)=linspace(1,1,index.n_elem);
    gamma_I.col(k-1)=tmp_vec;
  };
  
  E_W=linspace(0.5,0.5,ind(K+1)+1);

  timer.step("initialization");

  //*************************************************************
  double diff_m, sum_var, sum_mean, logit;
  vec old_m, c_W, y_0, y_k, E_W_0, E_W_k, gamma_I_k, m_betak_k;
  mat m2_beta0, m2_betak, X_0, X_k, X_m2_beta0, X_m2_betak;
  uvec order_beta0, order_betak, index_beta0, index_betak;
  //*************************************************************
  // CAVI
  diff_m=10;
	int iter_num = 0;
	
	X_0 = X.rows(ind(0),ind(1)); y_0 = y.subvec(ind(0),ind(1));
  while (diff_m > threshold){
    old_m = m_beta0;
    iter_num = iter_num+1;
    if (iter_num > max_iter){
      break;
    };
    
    // beta0: m_beta0, V_beta0
    // Z: gamma_Z
    E_W_0 = E_W.subvec(ind(0),ind(1));
    order_beta0 = sort_index(abs(m_beta0), "descending");
    for (int j = 0; j < p; ++j){
      uword j_ord = order_beta0(j);
      index_beta0 = find(linspace(0,p-1,p) != j_ord);
      sum_var = sum(X_0.col(j_ord).t()*diagmat(E_W_0)*X_0.col(j_ord))+pow(eta,-2);
      sum_mean = sum((y_0-0.5).t()*X_0.col(j_ord))-
        sum(X_0.col(j_ord).t()*diagmat(E_W_0)*X_0.cols(index_beta0)*
          (gamma_Z(index_beta0)%m_beta0(index_beta0)));
      for (int k=1; k<K+1; ++k){
        X_k = X.rows(ind(k)+1,ind(k+1)); y_k = y.subvec(ind(k)+1,ind(k+1));
        E_W_k = E_W.subvec(ind(k)+1,ind(k+1));
        gamma_I_k = gamma_I.col(k-1); m_betak_k = m_betak.col(k-1);
        sum_var = sum_var+
          sum(X_k.col(j_ord).t()*diagmat(E_W_k)*X_k.col(j_ord))*gamma_I_k(j_ord)+
          pow(tau(k-1),-2)*(1-gamma_I_k(j_ord));
        sum_mean = sum_mean+
          gamma_I_k(j_ord)*(sum((y_k-0.5).t()*X_k.col(j_ord))-
                              sum(X_k.col(j_ord).t()*diagmat(E_W_k)*X_k.cols(index_beta0)*
                                (gamma_I_k(index_beta0)%gamma_Z(index_beta0)%m_beta0(index_beta0)+
                                (1-gamma_I_k(index_beta0))%m_betak_k(index_beta0))))+
          (1-gamma_I_k(j_ord))*pow(tau(k-1),-2)*m_betak_k(j_ord);
      };
      V_beta0(j_ord) = 1/sum_var;
      m_beta0(j_ord) = sum_mean/sum_var;

      logit = pow(m_beta0(j_ord),2)/(2*V_beta0(j_ord))+log(q_0*sqrt(V_beta0(j_ord))/((1-q_0)*eta));
      gamma_Z(j_ord) = 1/(1+exp(-logit));
    };
    
    
    // betak: m_betak, V_betak
    // I_betak: gamma_I_k
    for (int k=1; k<K+1; ++k){
      X_k = X.rows(ind(k)+1,ind(k+1)); y_k = y.subvec(ind(k)+1,ind(k+1));
      E_W_k = E_W.subvec(ind(k)+1,ind(k+1));
      order_betak = sort_index(abs(m_betak_k), "descending");
      for (int j = 0; j < p; ++j){
        uword j_ord = order_betak(j);
        index_betak = find(linspace(0,p-1,p) != j_ord);
        gamma_I_k = gamma_I.col(k-1); m_betak_k = m_betak.col(k-1);
        V_betak(j_ord,k-1) = 1/(sum(X_k.col(j_ord).t()*diagmat(E_W_k)*X_k.col(j_ord))+pow(tau(k-1),-2));
        m_betak(j_ord,k-1) = (sum((y_k-0.5).t()*X_k.col(j_ord))-
                              sum(X_k.col(j_ord).t()*diagmat(E_W_k)*X_k.cols(index_betak)*
                                (gamma_I_k(index_betak)%gamma_Z(index_betak)%m_beta0(index_betak)+
                                  (1-gamma_I_k(index_betak))%m_betak_k(index_betak)))+
                              pow(tau(k-1),-2)*gamma_Z(j_ord)*m_beta0(j_ord))*V_betak(j_ord,k-1);
        
        m_betak_k = m_betak.col(k-1);
        logit = -pow(m_betak_k(j_ord),2)/(2*V_betak(j_ord,k-1))-log((1-q_k)*sqrt(V_betak(j_ord,k-1))/(q_k*tau(k-1)))+
          gamma_Z(j_ord)*(sum((y_k-0.5).t()*X_k.col(j_ord))*m_beta0(j_ord)-
                          0.5*(sum(X_k.col(j_ord).t()*diagmat(E_W_k)*X_k.col(j_ord))-pow(tau(k-1),-2))*
                                (pow(m_beta0(j_ord),2)+V_beta0(j_ord))-
                          m_beta0(j_ord)*sum(X_k.col(j_ord).t()*diagmat(E_W_k)*X_k.cols(index_betak)*
                                            (gamma_I_k(index_betak)%gamma_Z(index_betak)%m_beta0(index_betak)+
                                              (1-gamma_I_k(index_betak))%m_betak_k(index_betak))));
        gamma_I(j_ord,k-1) = 1/(1+exp(-logit));
      };
    };
    
    
    // omega
    // c_omega_0
    m2_beta0 = diagmat(gamma_Z%V_beta0)+
      (gamma_Z*gamma_Z.t()+diagmat(gamma_Z%(1-gamma_Z)))%(m_beta0*m_beta0.t());
    X_m2_beta0 = X_0*m2_beta0*X_0.t();
    c_W = sqrt(X_m2_beta0.diag());
    for (int i = 0; i < ind(1)+1; ++i){
      E_W(i) = 1.0/(2*c_W(i))*tanh(c_W(i)/2);
    };
    // c_omega_k
    for (int k=1; k<K+1; ++k){
      X_k = X.rows(ind(k)+1,ind(k+1));
      m2_betak = diagmat((1-gamma_I.col(k-1))%V_betak.col(k-1))+
        (gamma_I.col(k-1)*gamma_I.col(k-1).t()+diagmat(gamma_I.col(k-1)%(1-gamma_I.col(k-1))))%m2_beta0+
        ((1-gamma_I.col(k-1))*(1-gamma_I.col(k-1)).t()+diagmat((1-gamma_I.col(k-1))%gamma_I.col(k-1)))%(m_betak.col(k-1)*m_betak.col(k-1).t())+
        (gamma_I.col(k-1)*(1-gamma_I.col(k-1)).t()-diagmat(gamma_I.col(k-1)%(1-gamma_I.col(k-1))))%((gamma_Z%m_beta0)*m_betak.col(k-1).t())+
        ((1-gamma_I.col(k-1))*gamma_I.col(k-1).t()-diagmat((1-gamma_I.col(k-1))%gamma_I.col(k-1)))%(m_betak.col(k-1)*(gamma_Z%m_beta0).t());
      X_m2_betak = X_k*m2_betak*X_k.t();
      c_W = sqrt(X_m2_betak.diag());
      for (int i = 0; i < ind(k+1)-ind(k); ++i){
        E_W(ind(k)+1+i) = 1.0/(2*c_W(i))*tanh(c_W(i)/2);
      };
    };
    
    diff_m = 0;
    for (int j = 0; j < p; ++j){
      diff_m = diff_m+pow(m_beta0(j)-old_m(j),2);
    };

  };
	
  timer.step("CAVI");
  

	
	timer.step("end");
	

	return List::create(
		_["m_beta0"] = m_beta0,
		_["gamma_Z"] = gamma_Z,
    _["m_betak"] = m_betak,
    _["gamma_I"] = gamma_I,
    _["V_beta0"] = V_beta0,
    _["V_betak"] = V_betak,
    _["timer"] = timer
	);
};


// sample_ind
vec sample_ind_C(vec n_vec, int num_K) {
  // The indices of different source data sets
  vec indices(num_K+1);
  int start, end;

  indices(1)=0;
  for (int k = 0; k < num_K; ++k) {
    if (k == 0) {
      indices(k+1) = n_vec(k)-1;
    } else if (k == 1) {
      indices(k+1) = sum(n_vec.subvec(0,1))-1;
    } else {
      indices(k+1) = sum(n_vec.subvec(0,k))-1;
    }
  }
  return indices;
}

// lasso_est
vec lasso_est(mat X, vec y, double lambda){
  Function lasso_est_R("lasso_est_R", Environment::global_env());
  NumericVector tmp = lasso_est_R(X, y, lambda);
  vec res = tmp;
  return res;
}


