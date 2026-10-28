// Route B: multi-group renewal model fitted to (reported) cases by group, with the
// next-generation matrix shape G estimated and normalised to spectral radius 1, so
// that R is the long-run reproduction number. Options:
//   structured = 1: G = diag(a) C diag(b), with known contact matrices C (one per
//                   period) and estimated multipliers a, b (a[1] = b[1] = 1);
//   structured = 0: G free (all entries estimated), one matrix per period.
//   rw = 1: log R follows a random walk across weekly blocks; rw = 0: independent blocks.
// Constant group-specific reporting only changes G by a similarity transform, which
// the multipliers absorb (a / q, b * q), so R is unaffected.
data {
  int<lower=1> T;
  int<lower=1> N;
  int<lower=1> S;
  array[T, N] int<lower=0> Y;
  matrix[N, S] w;
  int<lower=1> B;
  array[T] int<lower=1, upper=B> block;
  int<lower=1> nP;
  array[T] int<lower=1, upper=nP> period;
  int<lower=0, upper=1> structured;
  array[nP] matrix[N, N] C;
  int<lower=0, upper=1> rw;
}
transformed data {
  matrix[T, N] Lam = rep_matrix(0, T, N);
  for (t in 2:T) for (i in 1:N) for (s in 1:min(S, t - 1)) Lam[t, i] += w[i, s] * Y[t - s, i];
}
parameters {
  vector<lower=0>[structured ? N - 1 : 0] a_raw;
  vector<lower=0>[structured ? N - 1 : 0] b_raw;
  array[structured ? 0 : nP] matrix<lower=0>[N, N] Gfree;
  vector<lower=0>[rw ? 0 : B] R_blk;
  real logR1;
  vector[rw ? B - 1 : 0] z;
  real<lower=0> tau;
  real<lower=0> inv_phi;
}
transformed parameters {
  vector[B] R;
  array[nP] matrix[N, N] Gn;
  if (rw) {
    vector[B] lr;
    lr[1] = logR1;
    for (b in 2:B) lr[b] = lr[b - 1] + tau * z[b - 1];
    R = exp(lr);
  } else {
    R = R_blk;
  }
  for (k in 1:nP) {
    matrix[N, N] G;
    if (structured) {
      vector[N] a = append_row(1, a_raw);
      vector[N] b = append_row(1, b_raw);
      G = diag_pre_multiply(a, diag_post_multiply(C[k], b));
    } else {
      G = Gfree[k];
    }
    // reject extreme proposals before the eigen-decomposition (non-finite input can crash it)
    if (is_inf(sum(G)) || is_nan(sum(G)) || sum(G) <= 0) reject("G not finite and positive");
    Gn[k] = G / max(abs(eigenvalues(G)));
  }
}
model {
  a_raw ~ lognormal(0, 1);
  b_raw ~ lognormal(0, 1);
  for (k in 1:(structured ? 0 : nP)) to_vector(Gfree[k]) ~ lognormal(0, 1);
  R_blk ~ gamma(1, 0.2);
  logR1 ~ normal(0, 1);
  z ~ std_normal();
  tau ~ normal(0, 0.2);
  inv_phi ~ normal(0, 1);
  for (t in 2:T) {
    vector[N] mu = R[block[t]] * (Gn[period[t]]' * Lam[t]');
    for (j in 1:N) Y[t, j] ~ neg_binomial_2(fmax(mu[j], 1e-8), 1 / inv_phi);
  }
}
