// Multi-group renewal model for weekly cases by age group (Denmark), a weekly-data version of
// stan/england.stan without the infection survey:
//   - contact matrices from CoMix (one per week, from the nearest survey round),
//   - reporting relative to week 1, by group and week, from
//       log q[j, k] = beta * lt[j, k]  (test_cov = 1: lt = log people tested per head, relative
//                                       to week 1; beta = how detection scales with testing)
//                   + random walk       (est_q = 1),
//   - optionally (adapt = 1) each group's susceptibility and infectiousness follow a weekly random
//     walk (log A, split by rho), which can pick up vaccination, immunity and unobserved changes
//     in contacts.
// Reconstructed infections use a smooth daily version of the weekly cases (Xd), and the
// likelihood is on the weekly counts:
//   X_hat[t, i] = Xd[t, i] / q[i, week(t)]
//   mu[t, j]    = q[j, week(t)] * R[week(t)] * sum_i Gn[i, j] sum_s w[s] X_hat[t - s, i]
//   Yw[k, j]   ~ NegBin(sum_{t in week k} mu[t, j], phi)
// With N = 1 it is a single-group model on total cases.
data {
  int<lower=1> T;                          // number of days
  int<lower=1> N;                          // number of groups
  int<lower=1> S;                          // longest generation time considered (days)
  int<lower=1> NW;                         // number of weeks
  array[T] int<lower=1, upper=NW> wk;      // week of each day
  int<lower=1, upper=NW> k0;               // first week in the likelihood (needs S days before it)
  matrix<lower=0>[T, N] Xd;                // smooth daily cases
  array[NW, N] int<lower=0> Yw;            // weekly cases
  matrix[N, S] w;                          // generation-time distribution of each group
  array[NW] matrix<lower=0>[N, N] C;       // contact matrix per week (rows: infectors)
  int<lower=0, upper=1> adapt;
  int<lower=0, upper=1> est_q;
  int<lower=0, upper=1> tau_by_group;
  int<lower=0, upper=1> test_cov;
  matrix[N, NW] lt;                        // log tests per head relative to week 1
}
parameters {
  vector<lower=0>[N - 1] a_raw;            // infectiousness / reporting, groups 2..N (group 1 = 1)
  vector<lower=0>[N - 1] b_raw;            // susceptibility x reporting, groups 2..N (group 1 = 1)
  real logR1;
  vector[NW - 1] z;
  real<lower=0> tau;
  real<lower=0> inv_phi;
  array[est_q] matrix[N, NW - 1] uz;
  array[est_q] vector<lower=0>[tau_by_group ? N : 1] tau_q;
  array[test_cov] real beta;
  matrix[adapt ? N - 1 : 0, adapt ? NW - 1 : 0] Az;
  array[adapt] real<lower=0> sigA;
  array[adapt] real<lower=0, upper=1> rho;
}
transformed parameters {
  vector[NW] R;
  matrix[N, NW] lq;                        // log reporting relative to week 1
  matrix[N, NW] logA;
  array[NW] matrix[N, N] Gn;
  matrix[T, N] Xhat;
  {
    vector[NW] lr;
    lr[1] = logR1;
    for (k in 2:NW) lr[k] = lr[k - 1] + tau * z[k - 1];
    R = exp(lr);
  }
  lq = test_cov ? beta[1] * lt : rep_matrix(0, N, NW);
  if (est_q) {
    vector[N] sd_q = tau_by_group ? tau_q[1] : rep_vector(tau_q[1][1], N);
    vector[N] u = rep_vector(0, N);
    for (k in 2:NW) {
      u += sd_q .* uz[1][, k - 1];
      lq[, k] += u;
    }
  }
  for (t in 1:T) for (i in 1:N) Xhat[t, i] = Xd[t, i] * exp(-lq[i, wk[t]]);
  {
    vector[N] a = append_row(1, a_raw);
    vector[N] b = append_row(1, b_raw);
    vector[N] lA = rep_vector(0, N);
    for (k in 1:NW) {
      if (adapt && k > 1) lA[2:N] += sigA[1] * Az[, k - 1];
      logA[, k] = lA;
      vector[N] sus = adapt ? exp(rho[1] * lA) : rep_vector(1, N);
      vector[N] inf = adapt ? exp((1 - rho[1]) * lA) : rep_vector(1, N);
      matrix[N, N] G = diag_pre_multiply(a .* inf, diag_post_multiply(C[k], b .* sus));
      if (is_inf(sum(G)) || is_nan(sum(G)) || sum(G) <= 0) reject("G not finite and positive");
      Gn[k] = G / max(abs(eigenvalues(G)));
    }
  }
}
model {
  a_raw ~ lognormal(0, 1);
  b_raw ~ lognormal(0, 1);
  logR1 ~ normal(0, 0.5);
  z ~ std_normal();
  tau ~ normal(0, 0.2);
  inv_phi ~ normal(0, 1);
  for (k in 1:est_q) {
    to_vector(uz[k]) ~ std_normal();
    tau_q[k] ~ normal(0, 0.3);
  }
  beta ~ normal(0.5, 0.5);
  to_vector(Az) ~ std_normal();
  sigA ~ normal(0, 0.15);
  rho ~ beta(2, 2);
  {
    matrix[NW, N] muw = rep_matrix(0, NW, N);
    for (t in 1:T) {
      if (wk[t] < k0) continue;
      vector[N] Lam = rep_vector(0, N);
      for (i in 1:N) for (s in 1:S) Lam[i] += w[i, s] * Xhat[t - s, i];
      muw[wk[t]] += (exp(lq[, wk[t]]) .* (R[wk[t]] * (Gn[wk[t]]' * Lam)))';
    }
    for (k in k0:NW) for (j in 1:N) Yw[k, j] ~ neg_binomial_2(fmax(muw[k, j], 1e-8), 1 / inv_phi);
  }
}
