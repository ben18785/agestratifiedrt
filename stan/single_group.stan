// Single-group renewal model fitted to total cases: the standard approach that
// ignores population structure. I_t ~ Poisson or NegBin with mean R_b(t) * sum_s w_s I_{t-s}.
data {
  int<lower=1> T;                          // number of days
  int<lower=1> S;                          // longest generation time considered (days)
  array[T] int<lower=0> I;                 // total reported cases on each day
  vector[S] w;                             // generation-time distribution: w[s] = P(s days)
  int<lower=1> B;                          // number of blocks (weeks) over which R is constant
  array[T] int<lower=1, upper=B> block;    // block that each day belongs to
  int<lower=0, upper=1> nb;                // 1: negative binomial (overdispersed), 0: Poisson
}
transformed data {
  // Lam[t]: infection pressure on day t, sum over s of w[s] * I[t - s]
  vector[T] Lam = rep_vector(0, T);
  for (t in 2:T) for (s in 1:min(S, t - 1)) Lam[t] += w[s] * I[t - s];
}
parameters {
  vector<lower=0>[B] R;                    // reproduction number in each block
  array[nb] real<lower=0> inv_phi;         // negative-binomial overdispersion, if nb = 1
}
model {
  R ~ gamma(1, 0.2);
  if (nb == 1) {
    inv_phi[1] ~ normal(0, 1);
    for (t in 2:T) I[t] ~ neg_binomial_2(R[block[t]] * Lam[t], 1 / inv_phi[1]);
  } else {
    for (t in 2:T) I[t] ~ poisson(R[block[t]] * Lam[t]);
  }
}
