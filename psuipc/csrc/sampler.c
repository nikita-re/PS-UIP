/* psuipc/csrc/sampler.c
 * =====================
 *
 * C port of psuipc/fast_sampler.py `_gibbs`, statistically equivalent to the
 * numba ("fast") backend. Single exported function `gibbs`. Dense linear algebra
 * (Cholesky + triangular solves) for the small q x q SPD system is implemented
 * locally; q <= 8. RNG is xoshiro256** seeded by splitmix64.
 *
 * The math, adaptation schedule, prior construction, PG augmentation, sigma MH,
 * M MH and Dirichlet-RW w MH mirror fast_sampler.py line for line. Bit-identical
 * RNG is NOT expected; statistical equivalence within Monte Carlo error IS.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PI M_PI
#define TRUNC 0.64
#define QMAX 8
#define NSWEEP 8

/* --------------------------------------------------------------------------- */
/* RNG: xoshiro256** seeded by splitmix64.                                      */
/* --------------------------------------------------------------------------- */
typedef struct {
    uint64_t s[4];
    /* cached spare standard normal (Box-Muller produces two at a time) */
    int has_spare;
    double spare;
} rng_t;

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static void rng_seed(rng_t *r, uint64_t seed) {
    uint64_t sm = seed;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&sm);
    r->has_spare = 0;
    r->spare = 0.0;
    /* warm up a few draws */
    for (int i = 0; i < 16; i++) {
        uint64_t s0 = r->s[0], s1 = r->s[1], s2 = r->s[2], s3 = r->s[3];
        (void)s0; (void)s1; (void)s2; (void)s3;
        uint64_t t = r->s[1] << 17;
        r->s[2] ^= r->s[0];
        r->s[3] ^= r->s[1];
        r->s[1] ^= r->s[2];
        r->s[0] ^= r->s[3];
        r->s[2] ^= t;
        r->s[3] = rotl(r->s[3], 45);
    }
}

static inline uint64_t rng_next(rng_t *r) {
    const uint64_t result = rotl(r->s[1] * 5, 7) * 9;
    const uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl(r->s[3], 45);
    return result;
}

/* uniform in (0,1): use top 53 bits, strictly in (0,1). */
static inline double rng_uniform(rng_t *r) {
    uint64_t x = rng_next(r);
    /* 53-bit mantissa; map to [0,1). Add tiny floor to avoid exact 0. */
    double u = (double)(x >> 11) * (1.0 / 9007199254740992.0);
    if (u <= 0.0) u = 2.2250738585072014e-308;
    if (u >= 1.0) u = 1.0 - 1e-16;
    return u;
}

/* standard normal via Box-Muller (cache the spare). */
static inline double rng_normal(rng_t *r) {
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    double rad = sqrt(-2.0 * log(u1));
    double ang = 2.0 * PI * u2;
    r->spare = rad * sin(ang);
    r->has_spare = 1;
    return rad * cos(ang);
}

/* exponential(1): -log(u). */
static inline double rng_exponential(rng_t *r) {
    return -log(rng_uniform(r));
}

/* gamma(shape a, scale 1) via Marsaglia-Tsang, with a<1 boost. */
static double rng_gamma(rng_t *r, double a) {
    if (a <= 0.0) return 0.0;
    if (a < 1.0) {
        double u = rng_uniform(r);
        return rng_gamma(r, a + 1.0) * pow(u, 1.0 / a);
    }
    double d = a - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = rng_normal(r);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = rng_uniform(r);
        double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
        if (log(u) < 0.5 * x2 + d * (1.0 - v + log(v))) return d * v;
    }
}

/* --------------------------------------------------------------------------- */
/* Polya-Gamma PG(1, z) sampler (Devroye method; mirror fast_sampler.py).        */
/* --------------------------------------------------------------------------- */
static inline double log_phi(double x) {
    return log(0.5 * erfc(-x / sqrt(2.0)));
}

static double mass_texpon(double z) {
    double t = TRUNC;
    double fz = 0.125 * PI * PI + 0.5 * z * z;
    double b = sqrt(1.0 / t) * (t * z - 1.0);
    double a = -sqrt(1.0 / t) * (t * z + 1.0);
    double x0 = log(fz) + fz * t;
    double xb = x0 - z + log_phi(b);
    double xa = x0 + z + log_phi(a);
    double qdivp = 4.0 / PI * (exp(xb) + exp(xa));
    return 1.0 / (1.0 + qdivp);
}

static double a_coef(int n, double x) {
    double k = (n + 0.5) * PI;
    if (x > TRUNC) {
        return k * exp(-0.5 * k * k * x);
    } else if (x > 0.0) {
        return k * pow(2.0 / (PI * x), 1.5) * exp(-2.0 * (n + 0.5) * (n + 0.5) / x);
    }
    return 0.0;
}

static double rtigauss(rng_t *r, double z) {
    z = fabs(z);
    double t = TRUNC;
    double x = t + 1.0;
    if ((1.0 / z) > t) {
        double alpha = 0.0;
        while (rng_uniform(r) > alpha) {
            double e1 = rng_exponential(r);
            double e2 = rng_exponential(r);
            while (e1 * e1 > 2.0 * e2 / t) {
                e1 = rng_exponential(r);
                e2 = rng_exponential(r);
            }
            x = t / ((1.0 + t * e1) * (1.0 + t * e1));
            alpha = exp(-0.5 * z * z * x);
        }
    } else {
        double mu = 1.0 / z;
        x = t + 1.0;
        while (x > t) {
            double y = rng_normal(r);
            y = y * y;
            double half_mu = 0.5 * mu;
            double mu_y = mu * y;
            x = mu + half_mu * mu_y - half_mu * sqrt(4.0 * mu_y + mu_y * mu_y);
            if (rng_uniform(r) > mu / (mu + x)) {
                x = mu * mu / x;
            }
        }
    }
    return x;
}

static double pg1(rng_t *r, double zin) {
    double z = fabs(zin) * 0.5;
    double fz = 0.125 * PI * PI + 0.5 * z * z;
    for (;;) {
        double x;
        if (rng_uniform(r) < mass_texpon(z)) {
            x = TRUNC + rng_exponential(r) / fz;
        } else {
            x = rtigauss(r, z);
        }
        double s = a_coef(0, x);
        double y = rng_uniform(r) * s;
        int n = 0;
        for (;;) {
            n += 1;
            if (n % 2 == 1) {
                s -= a_coef(n, x);
                if (y <= s) return 0.25 * x;
            } else {
                s += a_coef(n, x);
                if (y > s) break;
            }
        }
    }
}

/* --------------------------------------------------------------------------- */
/* Dense linear algebra for the q x q SPD system. Column-major not needed; we   */
/* store small matrices row-major in flat arrays of length q*q.                  */
/* --------------------------------------------------------------------------- */

/* Cholesky: A = L L^T, lower-triangular L (row-major, q x q). Returns 0 on ok. */
static int chol(const double *A, double *L, int q) {
    for (int i = 0; i < q * q; i++) L[i] = 0.0;
    for (int i = 0; i < q; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i * q + j];
            for (int k = 0; k < j; k++) sum -= L[i * q + k] * L[j * q + k];
            if (i == j) {
                if (sum <= 0.0) return 1;
                L[i * q + j] = sqrt(sum);
            } else {
                L[i * q + j] = sum / L[j * q + j];
            }
        }
    }
    return 0;
}

/* Solve L x = b (forward), L lower-triangular row-major. */
static void fwd_solve(const double *L, const double *b, double *x, int q) {
    for (int i = 0; i < q; i++) {
        double sum = b[i];
        for (int k = 0; k < i; k++) sum -= L[i * q + k] * x[k];
        x[i] = sum / L[i * q + i];
    }
}

/* Solve L^T x = b (backward), L lower-triangular row-major. */
static void bwd_solve(const double *L, const double *b, double *x, int q) {
    for (int i = q - 1; i >= 0; i--) {
        double sum = b[i];
        for (int k = i + 1; k < q; k++) sum -= L[k * q + i] * x[k];
        x[i] = sum / L[i * q + i];
    }
}

/* Solve A x = b given chol(A)=L: forward then backward. */
static void chol_solve(const double *L, const double *b, double *x,
                       double *tmp, int q) {
    fwd_solve(L, b, tmp, q);
    bwd_solve(L, tmp, x, q);
}

/* --------------------------------------------------------------------------- */
/* The exported Gibbs sampler.                                                  */
/* --------------------------------------------------------------------------- */
EXPORT void gibbs(const double *D, const double *y, int n, int q,
                  int is_binary, int has_uip, int K,
                  const double *muC, const double *I_U, const double *gamma,
                  double M_lo, double M_hi,
                  double beta0_mean0, double beta0_prec0,
                  double weak_theta_var, double b_var,
                  int n_draws, int n_tune, unsigned long long seed,
                  double *theta_out, double *M_out, double *w_out) {
    rng_t rng;
    rng_seed(&rng, (uint64_t)seed);

    /* DtD (q x q row-major), Dty (q), kappa = y - 0.5 (n). */
    double *DtD = (double *)malloc(sizeof(double) * q * q);
    double *Dty = (double *)malloc(sizeof(double) * q);
    double *kappa = (double *)malloc(sizeof(double) * n);
    for (int a = 0; a < q; a++) {
        Dty[a] = 0.0;
        for (int b = 0; b < q; b++) DtD[a * q + b] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        const double *Di = D + (size_t)i * q;
        kappa[i] = y[i] - 0.5;
        for (int a = 0; a < q; a++) {
            Dty[a] += Di[a] * y[i];
            double dia = Di[a];
            for (int b = 0; b < q; b++) DtD[a * q + b] += dia * Di[b];
        }
    }

    /* Prior diagonal precisions / means for phi. */
    double P0[QMAX], m0[QMAX];
    P0[0] = 0.0;
    P0[1] = 1.0 / weak_theta_var;
    for (int j = 2; j < q; j++) P0[j] = 1.0 / b_var;
    for (int j = 0; j < q; j++) m0[j] = 0.0;

    /* Working scratch. */
    double Lam[QMAX * QMAX], L[QMAX * QMAX];
    double rhs[QMAX], phi[QMAX], mean[QMAX], zvec[QMAX], noise[QMAX], tmp[QMAX];
    double *omega = (double *)malloc(sizeof(double) * n);
    double *eta = (double *)malloc(sizeof(double) * n);

    /* Initialise phi by ridge-regularised least squares on the working response.
     * A0 = DtD + I; rhs0 = D' yw, yw = kappa (binary) or y (continuous). */
    {
        double A0[QMAX * QMAX], rhs0[QMAX];
        for (int a = 0; a < q; a++) {
            for (int b = 0; b < q; b++) A0[a * q + b] = DtD[a * q + b];
            A0[a * q + a] += 1.0;
            rhs0[a] = 0.0;
        }
        for (int i = 0; i < n; i++) {
            const double *Di = D + (size_t)i * q;
            double yw = is_binary ? kappa[i] : y[i];
            for (int a = 0; a < q; a++) rhs0[a] += Di[a] * yw;
        }
        if (chol(A0, L, q) != 0) {
            /* fall back: heavier ridge */
            for (int a = 0; a < q; a++) A0[a * q + a] += 10.0;
            chol(A0, L, q);
        }
        chol_solve(L, rhs0, phi, tmp, q);
    }

    /* sigma init: residual RMS. */
    double sigma;
    {
        double ssr = 0.0;
        for (int i = 0; i < n; i++) {
            const double *Di = D + (size_t)i * q;
            double dp = 0.0;
            for (int a = 0; a < q; a++) dp += Di[a] * phi[a];
            double rr = y[i] - dp;
            ssr += rr * rr;
        }
        int denom = (n - q) > 1 ? (n - q) : 1;
        sigma = sqrt(ssr / denom);
        if (sigma < 1e-2) sigma = 1e-2;
    }

    /* w init = gamma / sum(gamma); M init = mid. */
    double *w = (double *)malloc(sizeof(double) * (K > 0 ? K : 1));
    double *wp = (double *)malloc(sizeof(double) * (K > 0 ? K : 1));
    double *a_prop = (double *)malloc(sizeof(double) * (K > 0 ? K : 1));
    double *a_back = (double *)malloc(sizeof(double) * (K > 0 ? K : 1));
    {
        double gsum = 0.0;
        for (int k = 0; k < K; k++) gsum += gamma[k];
        for (int k = 0; k < K; k++) w[k] = gamma[k] / gsum;
    }
    double M = 0.5 * (M_lo + M_hi);
    if (M > M_hi) M = M_hi;
    if (M < M_lo) M = M_lo;

    /* Adaptive RW step sizes. */
    double step_sig = 0.2, step_M = 0.5, conc_w = 200.0;
    double acc_sig = 0.0, acc_M = 0.0, acc_w = 0.0, tot = 0.0;

    for (int k = 0; k < K; k++) w_out[k] = 0.0;

    int total = n_tune + n_draws;
    for (int it = 0; it < total; it++) {
        /* ---- beta0 prior from (w, M) ---- */
        if (has_uip) {
            double sW = 0.0, muW = 0.0;
            for (int k = 0; k < K; k++) {
                sW += w[k] * I_U[k];
                muW += w[k] * muC[k];
            }
            P0[0] = M * sW;
            m0[0] = muW;
        } else {
            P0[0] = beta0_prec0;
            m0[0] = beta0_mean0;
        }

        /* ---- phi update (Gaussian block) ---- */
        if (is_binary) {
            /* eta = D phi */
            for (int i = 0; i < n; i++) {
                const double *Di = D + (size_t)i * q;
                double e = 0.0;
                for (int a = 0; a < q; a++) e += Di[a] * phi[a];
                if (e > 30.0) e = 30.0;
                else if (e < -30.0) e = -30.0;
                eta[i] = e;
            }
            for (int i = 0; i < n; i++) {
                double om = pg1(&rng, eta[i]);
                if (om < 1e-6) om = 1e-6;
                omega[i] = om;
            }
            /* Lam = D' diag(omega) D ; rhs = D' kappa */
            for (int a = 0; a < q; a++) {
                rhs[a] = 0.0;
                for (int b = 0; b < q; b++) Lam[a * q + b] = 0.0;
            }
            for (int i = 0; i < n; i++) {
                const double *Di = D + (size_t)i * q;
                double oi = omega[i];
                for (int a = 0; a < q; a++) {
                    double oda = oi * Di[a];
                    rhs[a] += Di[a] * kappa[i];
                    for (int b = 0; b < q; b++) Lam[a * q + b] += oda * Di[b];
                }
            }
        } else {
            double inv_s2 = 1.0 / (sigma * sigma);
            for (int a = 0; a < q; a++) {
                rhs[a] = Dty[a] * inv_s2;
                for (int b = 0; b < q; b++) Lam[a * q + b] = DtD[a * q + b] * inv_s2;
            }
        }
        for (int j = 0; j < q; j++) {
            Lam[j * q + j] += P0[j];
            rhs[j] += P0[j] * m0[j];
        }
        /* L = chol(Lam); mean = solve(Lam, rhs); z ~ N(0,I); noise = solve(L^T, z) */
        chol(Lam, L, q);
        chol_solve(L, rhs, mean, tmp, q);
        for (int a = 0; a < q; a++) zvec[a] = rng_normal(&rng);
        bwd_solve(L, zvec, noise, q);
        for (int a = 0; a < q; a++) phi[a] = mean[a] + noise[a];

        /* ---- sigma update (continuous), MH on log sigma ---- */
        if (!is_binary) {
            double ssr = 0.0;
            for (int i = 0; i < n; i++) {
                const double *Di = D + (size_t)i * q;
                double dp = 0.0;
                for (int a = 0; a < q; a++) dp += Di[a] * phi[a];
                double rr = y[i] - dp;
                ssr += rr * rr;
            }
            double ls = log(sigma);
            double lp = -n * ls - 0.5 * ssr / (sigma * sigma)
                        - 0.125 * sigma * sigma + ls;
            double ls_p = ls + step_sig * rng_normal(&rng);
            double sp = exp(ls_p);
            double lp_p = -n * ls_p - 0.5 * ssr / (sp * sp)
                          - 0.125 * sp * sp + ls_p;
            if (log(rng_uniform(&rng) + 1e-300) < lp_p - lp) {
                sigma = sp;
                if (it < n_tune) acc_sig += 1.0;
            }
        }

        /* ---- M and w updates (UIP only). NSWEEP inner sweeps. ---- */
        double b0 = phi[0];
        int nsw = has_uip ? NSWEEP : 0;
        for (int sweep = 0; sweep < nsw; sweep++) {
            double sW = 0.0, muW = 0.0;
            for (int k = 0; k < K; k++) {
                sW += w[k] * I_U[k];
                muW += w[k] * muC[k];
            }
            double d2 = (b0 - muW) * (b0 - muW);
            /* M: target 1.5*logM - 0.5*M*sW*d2, truncated (M_lo, M_hi) */
            double lM = log(M);
            double lM_p = lM + step_M * rng_normal(&rng);
            double Mp = exp(lM_p);
            if (M_lo < Mp && Mp < M_hi) {
                double cur = 1.5 * lM - 0.5 * M * sW * d2;
                double prop = 1.5 * lM_p - 0.5 * Mp * sW * d2;
                if (log(rng_uniform(&rng) + 1e-300) < prop - cur) {
                    M = Mp;
                    if (it < n_tune) acc_M += 1.0;
                }
            }

            /* w: Dirichlet RW proposal, target Dir(gamma)*N(beta0|muW,1/(M sW)) */
            if (K >= 2) {
                for (int k = 0; k < K; k++) a_prop[k] = conc_w * w[k] + 1e-6;
                double gtot = 0.0;
                for (int k = 0; k < K; k++) {
                    wp[k] = rng_gamma(&rng, a_prop[k]);
                    gtot += wp[k];
                }
                for (int k = 0; k < K; k++) wp[k] = wp[k] / gtot;
                double sWp = 0.0, muWp = 0.0;
                for (int k = 0; k < K; k++) {
                    sWp += wp[k] * I_U[k];
                    muWp += wp[k] * muC[k];
                }
                double lt_cur = 0.0, lt_prop = 0.0;
                for (int k = 0; k < K; k++) {
                    lt_cur += (gamma[k] - 1.0) * log(w[k] + 1e-300);
                    lt_prop += (gamma[k] - 1.0) * log(wp[k] + 1e-300);
                }
                double precc = M * sW;
                double precp = M * sWp;
                lt_cur += 0.5 * log(precc) - 0.5 * precc * (b0 - muW) * (b0 - muW);
                lt_prop += 0.5 * log(precp) - 0.5 * precp * (b0 - muWp) * (b0 - muWp);
                /* proposal correction q(w|wp)/q(wp|w), Dirichlet densities */
                for (int k = 0; k < K; k++) a_back[k] = conc_w * wp[k] + 1e-6;
                double lq_fwd = 0.0, lq_bwd = 0.0, sfa = 0.0, sba = 0.0;
                for (int k = 0; k < K; k++) {
                    sfa += a_prop[k];
                    sba += a_back[k];
                }
                lq_fwd += lgamma(sfa);
                lq_bwd += lgamma(sba);
                for (int k = 0; k < K; k++) {
                    lq_fwd += -lgamma(a_prop[k]) + (a_prop[k] - 1.0) * log(wp[k] + 1e-300);
                    lq_bwd += -lgamma(a_back[k]) + (a_back[k] - 1.0) * log(w[k] + 1e-300);
                }
                if (log(rng_uniform(&rng) + 1e-300) < (lt_prop - lt_cur) + (lq_bwd - lq_fwd)) {
                    for (int k = 0; k < K; k++) w[k] = wp[k];
                    if (it < n_tune) acc_w += 1.0;
                }
            }
        }

        /* ---- adapt during warmup ---- */
        if (it < n_tune) {
            tot += 1.0;
            if ((it + 1) % 50 == 0 && tot > 0.0) {
                if (!is_binary) {
                    double ar = acc_sig / tot;
                    step_sig *= (ar > 0.4) ? 1.15 : ((ar < 0.25) ? 0.85 : 1.0);
                }
                if (has_uip) {
                    double denom = tot * NSWEEP;
                    double arM = acc_M / denom;
                    step_M *= (arM > 0.4) ? 1.15 : ((arM < 0.25) ? 0.85 : 1.0);
                    if (K >= 2) {
                        double arw = acc_w / denom;
                        if (arw > 0.4) {
                            conc_w = (conc_w * 0.8 < 8000.0) ? conc_w * 0.8 : 8000.0;
                        } else if (arw < 0.2) {
                            conc_w = (conc_w * 1.25 > 20.0) ? conc_w * 1.25 : 20.0;
                        }
                    }
                }
                acc_sig = 0.0;
                acc_M = 0.0;
                acc_w = 0.0;
                tot = 0.0;
            }
        } else {
            int i_store = it - n_tune;
            theta_out[i_store] = phi[1];
            M_out[i_store] = M;
            for (int k = 0; k < K; k++) w_out[k] += w[k];
        }
    }

    if (n_draws > 0) {
        for (int k = 0; k < K; k++) w_out[k] /= n_draws;
    }

    free(DtD); free(Dty); free(kappa);
    free(omega); free(eta);
    free(w); free(wp); free(a_prop); free(a_back);
}
