// Multi-group renewal model for reported cases by group, with reporting probabilities that
// can change over time, and an optional random-sample infection survey (like the ONS
// Infection Survey) that measures prevalence by group.
//
// Infections are reconstructed from reported cases, X_hat[t, i] = Y[t, i] / q[t, i], and
// E[Y[t, j]] = q[t, j] * R * sum_i Gn[i, j] * sum_s w[i, s] * X_hat[t - s, i].
// G = diag(a) C diag(b) with known contact matrices C (one per period), normalised to
// spectral radius 1, so R is the long-run reproduction number.
// Reporting: q[t, i] = exp(u[i, week]) (relative to week 1) when there is no infection survey,
// or inv_logit(lq0[i] + u[i, week]) when there is one (the survey fixes the absolute level).
// u[i, 1] = 0 and u follows a random walk across weeks. With est_q = 0, q is constant.
data {
  int<lower=1> T;                          // number of days
  int<lower=1> N;                          // number of groups
  int<lower=1> S;                          // longest generation time considered (days)
  array[T, N] int<lower=0> Y;              // reported cases by day and group
  matrix[N, S] w;                          // generation-time distribution of each group
  int<lower=1> B;                          // number of blocks (weeks) for R
  array[T] int<lower=1, upper=B> block;    // block of each day
  int<lower=1> nP;                         // number of contact periods
  array[T] int<lower=1, upper=nP> period;  // contact period of each day
  array[nP] matrix[N, N] C;                // contact matrix for each period (rows: infectors)
  int<lower=0, upper=1> est_q;             // 1: reporting varies by week; 0: constant
  int<lower=1> W;                          // number of weeks for reporting
  array[T] int<lower=1, upper=W> qweek;    // reporting week of each day
  int<lower=0, upper=1> ons;               // 1: use the infection survey
  vector<lower=0>[N] Npop;                 // population of each group (for prevalence)
  int<lower=1> D;                          // days for which an infection stays detectable
  vector<lower=0, upper=1>[D] d;           // d[u + 1] = P(still positive u days after infection)
  int<lower=0> K;                          // number of survey windows
  array[K] int<lower=D, upper=T> win_start;  // first day of each window
  array[K] int<lower=D, upper=T> win_end;    // last day of each window
  array[K, N] int<lower=0> n_tested;       // people tested per window and group
  array[K, N] int<lower=0> n_pos;          // positives per window and group
}
parameters {
  vector<lower=0>[N - 1] a_raw;            // infectiousness multipliers, groups 2..N (group 1 = 1)
  vector<lower=0>[N - 1] b_raw;            // susceptibility multipliers, groups 2..N (group 1 = 1)
  real logR1;                              // log R in the first block
  vector[B - 1] z;                         // standardised weekly steps in log R
  real<lower=0> tau;                       // sd of weekly steps in log R
  real<lower=0> inv_phi;                   // negative-binomial overdispersion
  array[est_q] matrix[N, W - 1] uz;        // standardised weekly steps in log (or logit) reporting
  array[est_q] real<lower=0> tau_q;        // sd of those steps
  vector[ons ? N : 0] lq0;                 // logit reporting in week 1, if the survey is used
}
transformed parameters {
  vector[B] R;                             // long-run reproduction number by block
  array[nP] matrix[N, N] Gn;               // normalised matrix shape by period
  matrix[T, N] q;                          // reporting probability (or relative reporting)
  matrix[T, N] Xhat;                       // reconstructed infections
  {
    vector[B] lr;
    lr[1] = logR1;
    for (b in 2:B) lr[b] = lr[b - 1] + tau * z[b - 1];
    R = exp(lr);
  }
  {
    vector[N] a = append_row(1, a_raw);
    vector[N] b = append_row(1, b_raw);
    for (k in 1:nP) {
      matrix[N, N] G = diag_pre_multiply(a, diag_post_multiply(C[k], b));
      if (is_inf(sum(G)) || is_nan(sum(G)) || sum(G) <= 0) reject("G not finite and positive");
      Gn[k] = G / max(abs(eigenvalues(G)));
    }
  }
  {
    matrix[N, W] u = rep_matrix(0, N, W);  // weekly log (or logit) reporting relative to week 1
    if (est_q) for (k in 2:W) u[, k] = u[, k - 1] + tau_q[1] * uz[1][, k - 1];
    for (t in 1:T) for (i in 1:N)
      q[t, i] = ons ? inv_logit(lq0[i] + u[i, qweek[t]]) : exp(u[i, qweek[t]]);
  }
  Xhat = to_matrix(Y) ./ q;
}
model {
  a_raw ~ lognormal(0, 1);
  b_raw ~ lognormal(0, 1);
  logR1 ~ normal(0, 1);
  z ~ std_normal();
  tau ~ normal(0, 0.2);
  inv_phi ~ normal(0, 1);
  for (k in 1:est_q) to_vector(uz[k]) ~ std_normal();
  tau_q ~ normal(0, 0.3);
  lq0 ~ normal(-1, 1.5);
  // cases
  for (t in 2:T) {
    vector[N] Lam = rep_vector(0, N);      // infection pressure from reconstructed infections
    for (i in 1:N) for (s in 1:min(S, t - 1)) Lam[i] += w[i, s] * Xhat[t - s, i];
    vector[N] mu = q[t]' .* (R[block[t]] * (Gn[period[t]]' * Lam));   // expected reported cases
    for (j in 1:N) Y[t, j] ~ neg_binomial_2(fmax(mu[j], 1e-8), 1 / inv_phi);
  }
  // infection survey: positives ~ binomial(tested, mean prevalence over the window)
  if (ons) {
    for (k in 1:K) for (j in 1:N) {
      real prev = 0;
      for (t in win_start[k]:win_end[k]) for (u in 0:(D - 1)) prev += d[u + 1] * Xhat[t - u, j] / Npop[j];
      prev /= (win_end[k] - win_start[k] + 1);
      n_pos[k, j] ~ binomial(n_tested[k, j], fmin(prev, 1 - 1e-9));
    }
  }
}
