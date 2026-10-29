// Multi-group renewal model for cases by age group in England, with
//   - contact matrices from CoMix (one per week, from two-week survey waves),
//   - vaccination as a known, time-varying reduction in susceptibility,
//   - group-specific day-of-week reporting effects,
//   - weekly reporting probabilities (optional), and
//   - the ONS Infection Survey (optional), which fixes the absolute level of reporting.
// With N = 1 it is a single-group renewal model on total cases.
//
// Reconstructed infections (on the specimen-date time scale):
//   X_hat[t, i] = Y[t, i] / (q[t, i] * 7 * dow[i, weekday(t)]).
// Expected cases:
//   mu[t, j] = q[t, j] * 7 * dow[j, weekday(t)] * R[block(t)] * sum_i Gn[i, j] * sum_s w[i, s] X_hat[t - s, i],
// with G = diag(a) C diag(b .* v) for the week's contact matrix C and susceptibility factor v
// (1 - vaccine protection), normalised to spectral radius 1, so R is the long-run reproduction
// number at that week's contacts and immunity. Optionally (deplete = 1, which needs the survey
// so that X_hat is on an absolute scale) v also falls with each group's infections since day 1:
// v_eff = v - (cumulative X_hat / population) / s0, with s0 the susceptible fraction on day 1.
data {
  int<lower=1> T;                          // number of days
  int<lower=1> N;                          // number of groups
  int<lower=1> S;                          // longest generation time considered (days)
  int<lower=S + 1, upper=T> t0;            // first day whose cases enter the likelihood
  array[T, N] int<lower=0> Y;              // cases by specimen date and group
  matrix[N, S] w;                          // generation-time distribution of each group
  int<lower=1> B;                          // number of blocks (weeks) for R
  array[T] int<lower=1, upper=B> block;    // block of each day
  int<lower=1> nP;                         // number of matrix periods (weeks)
  array[T] int<lower=1, upper=nP> period;  // matrix period of each day
  array[nP] matrix<lower=0>[N, N] C;       // contact matrix per period (rows: infectors)
  array[nP] vector<lower=0, upper=1>[N] v; // susceptibility factor from vaccination per period
  array[nP] int<lower=1, upper=T> pstart;  // first day of each period
  int<lower=0, upper=1> deplete;           // 1: susceptibility falls with infections in the window
  vector<lower=0, upper=1>[N] s0;          // susceptible fraction on day 1 (for deplete)
  array[T] int<lower=1, upper=7> dow;      // day of the week of each day
  int<lower=0, upper=1> est_q;             // 1: reporting varies by week; 0: constant
  int<lower=0, upper=1> tau_by_group;      // 1: each group has its own sd of weekly reporting steps
  int<lower=1> W;                          // number of weeks for reporting
  array[T] int<lower=1, upper=W> qweek;    // reporting week of each day
  int<lower=0, upper=1> ons;               // 1: use the infection survey
  vector<lower=0>[N] Npop;                 // population of each group
  int<lower=1> D;                          // days for which an infection can test positive
  vector<lower=0, upper=1>[D] d;           // d[u + 1] = P(PCR positive u days after infection)
  int<lower=0> delta;                      // days from infection to specimen date
  int<lower=0> K;                          // number of survey windows
  array[K] int<lower=1, upper=T> win_start;  // first day of each window
  array[K] int<lower=1, upper=T> win_end;    // last day of each window
  array[K, N] int<lower=0> n_tested;       // people tested per window and group
  array[K, N] int<lower=0> n_pos;          // positives per window and group
}
parameters {
  vector<lower=0>[N - 1] a_raw;            // infectiousness / reporting, groups 2..N (group 1 = 1)
  vector<lower=0>[N - 1] b_raw;            // susceptibility x reporting, groups 2..N (group 1 = 1)
  real logR1;                              // log R in the first block
  vector[B - 1] z;                         // standardised weekly steps in log R
  real<lower=0> tau;                       // sd of weekly steps in log R
  real<lower=0> inv_phi;                   // negative-binomial overdispersion
  array[N] simplex[7] dw;                  // day-of-week shares of each group's cases
  array[est_q] matrix[N, W - 1] uz;        // standardised weekly steps in log (or logit) reporting
  array[est_q] vector<lower=0>[tau_by_group ? N : 1] tau_q;  // sd of those steps (one, or one per group)
  vector[ons ? N : 0] lq0;                 // logit reporting in week 1, if the survey is used
}
transformed parameters {
  vector[B] R;                             // long-run reproduction number by block
  array[nP] matrix[N, N] Gn;               // normalised matrix shape by period
  matrix[T, N] q;                          // reporting probability (or relative reporting)
  matrix[T, N] rep;                        // q times the day-of-week effect
  matrix[T, N] Xhat;                       // reconstructed infections
  {
    vector[B] lr;
    lr[1] = logR1;
    for (b in 2:B) lr[b] = lr[b - 1] + tau * z[b - 1];
    R = exp(lr);
  }
  {
    matrix[N, W] u = rep_matrix(0, N, W);  // weekly log (or logit) reporting relative to week 1
    if (est_q) {
      vector[N] sd_q = tau_by_group ? tau_q[1] : rep_vector(tau_q[1][1], N);
      for (k in 2:W) u[, k] = u[, k - 1] + sd_q .* uz[1][, k - 1];
    }
    for (t in 1:T) for (i in 1:N) {
      q[t, i] = ons ? inv_logit(lq0[i] + u[i, qweek[t]]) : exp(u[i, qweek[t]]);
      rep[t, i] = q[t, i] * 7 * dw[i][dow[t]];
    }
  }
  Xhat = to_matrix(Y) ./ rep;
  {
    vector[N] a = append_row(1, a_raw);
    vector[N] b = append_row(1, b_raw);
    vector[N] cum = rep_vector(0, N);      // infections since day 1 as a fraction of the population
    int t_done = 1;
    for (k in 1:nP) {
      vector[N] veff = v[k];
      if (deplete) {
        for (t in t_done:(pstart[k] - 1)) cum += Xhat[t]' ./ Npop;
        t_done = pstart[k];
        veff = fmax(v[k] - cum ./ s0, 0.02);
      }
      matrix[N, N] G = diag_pre_multiply(a, diag_post_multiply(C[k], b .* veff));
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
  for (i in 1:N) dw[i] ~ dirichlet(rep_vector(10, 7));
  for (k in 1:est_q) {
    to_vector(uz[k]) ~ std_normal();
    tau_q[k] ~ normal(0, 0.3);
  }
  lq0 ~ normal(-1, 1.5);
  // cases
  for (t in t0:T) {
    vector[N] Lam = rep_vector(0, N);      // infection pressure from reconstructed infections
    for (i in 1:N) for (s in 1:S) Lam[i] += w[i, s] * Xhat[t - s, i];
    vector[N] mu = rep[t]' .* (R[block[t]] * (Gn[period[t]]' * Lam));
    for (j in 1:N) Y[t, j] ~ neg_binomial_2(fmax(mu[j], 1e-8), 1 / inv_phi);
  }
  // infection survey: positives ~ binomial(tested, mean prevalence over the window).
  // Prevalence on day t uses infections on day t - u, i.e. X_hat on day t - u + delta.
  if (ons) {
    for (k in 1:K) for (j in 1:N) {
      real prev = 0;
      for (t in win_start[k]:win_end[k])
        for (u in 0:(D - 1)) prev += d[u + 1] * Xhat[t - u + delta, j] / Npop[j];
      prev /= (win_end[k] - win_start[k] + 1);
      n_pos[k, j] ~ binomial(n_tested[k, j], fmin(prev, 1 - 1e-9));
    }
  }
}
generated quantities {
  // modelled ONS positivity (%) per window and group, for checking the fit
  array[K] vector[N] prev_pct;
  for (k in 1:K) for (j in 1:N) {
    real prev = 0;
    for (t in win_start[k]:win_end[k])
      for (u in 0:(D - 1)) prev += d[u + 1] * Xhat[t - u + delta, j] / Npop[j];
    prev_pct[k][j] = 100 * prev / (win_end[k] - win_start[k] + 1);
  }
}
