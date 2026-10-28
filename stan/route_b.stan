// Multi-group renewal model fitted to reported cases by group ("Route B" in the note).
// The shape G of the transmission matrix is estimated and normalised to spectral
// radius 1, so that R is the long-run reproduction number. Options:
//   structured = 1: G = diag(a) C diag(b), with known contact matrices C (one per
//                   period) and estimated multipliers a, b (a[1] = b[1] = 1);
//   structured = 0: G free (all entries estimated), one matrix per period.
//   rw = 1: log R follows a random walk across weekly blocks; rw = 0: independent blocks.
// Constant group-specific reporting only changes G by a similarity transform, which
// the multipliers absorb (a / q, b * q), so R is unaffected.
data {
  int<lower=1> T;                          // number of days
  int<lower=1> N;                          // number of groups
  int<lower=1> S;                          // longest generation time considered (days)
  array[T, N] int<lower=0> Y;              // reported cases by day (rows) and group (columns)
  matrix[N, S] w;                          // generation-time distribution of each group: w[i, s] = P(s days)
  int<lower=1> B;                          // number of blocks (weeks) over which R is constant
  array[T] int<lower=1, upper=B> block;    // block that each day belongs to
  int<lower=1> nP;                         // number of contact periods (e.g. 2: before and after a change)
  array[T] int<lower=1, upper=nP> period;  // contact period that each day belongs to
  int<lower=0, upper=1> structured;        // 1: G built from the contact matrices C; 0: G free
  array[nP] matrix[N, N] C;                // contact matrix for each period (rows: infectors); used only if structured = 1
  int<lower=0, upper=1> rw;                // 1: random walk on log R; 0: independent R per block
}
transformed data {
  // Lam[t, i]: infection pressure from group i on day t, computed from reported cases,
  // sum over s of w[i, s] * Y[t - s, i]
  matrix[T, N] Lam = rep_matrix(0, T, N);
  for (t in 2:T) for (i in 1:N) for (s in 1:min(S, t - 1)) Lam[t, i] += w[i, s] * Y[t - s, i];
}
parameters {
  vector<lower=0>[structured ? N - 1 : 0] a_raw;   // infectiousness multipliers, groups 2..N (group 1 fixed at 1); estimate infectiousness / reporting
  vector<lower=0>[structured ? N - 1 : 0] b_raw;   // susceptibility multipliers, groups 2..N (group 1 fixed at 1); estimate susceptibility x reporting
  array[structured ? 0 : nP] matrix<lower=0>[N, N] Gfree;  // free (unnormalised) matrix for each period, if structured = 0
  vector<lower=0>[rw ? 0 : B] R_blk;               // R in each block, if rw = 0
  real logR1;                                      // log R in the first block (used only if rw = 1)
  vector[rw ? B - 1 : 0] z;                        // standardised week-to-week steps in log R, if rw = 1
  real<lower=0> tau;                               // sd of the week-to-week steps in log R (used only if rw = 1)
  real<lower=0> inv_phi;                           // negative-binomial overdispersion (0 = Poisson)
}
transformed parameters {
  vector[B] R;                 // long-run reproduction number in each block
  array[nP] matrix[N, N] Gn;   // matrix shape in each period, normalised to spectral radius 1
  if (rw) {
    vector[B] lr;              // log R by block
    lr[1] = logR1;
    for (b in 2:B) lr[b] = lr[b - 1] + tau * z[b - 1];
    R = exp(lr);
  } else {
    R = R_blk;
  }
  for (k in 1:nP) {
    matrix[N, N] G;            // unnormalised matrix for period k (rows: infectors, columns: infectees)
    if (structured) {
      vector[N] a = append_row(1, a_raw);   // infectiousness multipliers, a[1] = 1
      vector[N] b = append_row(1, b_raw);   // susceptibility multipliers, b[1] = 1
      G = diag_pre_multiply(a, diag_post_multiply(C[k], b));   // diag(a) C diag(b)
    } else {
      G = Gfree[k];
    }
    // reject extreme proposals before the eigen-decomposition (non-finite input can crash it)
    if (is_inf(sum(G)) || is_nan(sum(G)) || sum(G) <= 0) reject("G not finite and positive");
    Gn[k] = G / max(abs(eigenvalues(G)));
  }
}
model {
  // priors (parameters unused under the chosen options are simply drawn from their priors)
  a_raw ~ lognormal(0, 1);
  b_raw ~ lognormal(0, 1);
  for (k in 1:(structured ? 0 : nP)) to_vector(Gfree[k]) ~ lognormal(0, 1);
  R_blk ~ gamma(1, 0.2);
  logR1 ~ normal(0, 1);
  z ~ std_normal();
  tau ~ normal(0, 0.2);
  inv_phi ~ normal(0, 1);
  // likelihood: expected cases in group j on day t are
  // R x (sum over infector groups i of Gn[i, j] x Lam[t, i])
  for (t in 2:T) {
    vector[N] mu = R[block[t]] * (Gn[period[t]]' * Lam[t]');   // expected cases by group on day t
    for (j in 1:N) Y[t, j] ~ neg_binomial_2(fmax(mu[j], 1e-8), 1 / inv_phi);
  }
}
