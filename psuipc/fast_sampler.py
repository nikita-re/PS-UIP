"""
psuipc/fast_sampler.py
======================

A hand-written, numba-compiled Gibbs sampler for the PS-UIP analysis model,
a drop-in replacement for the PyMC/NUTS sampling step in ``methods.py``. It
targets exactly the same posterior, so the two are statistically consistent (run
``methods.py`` with ``PSUIPC_BACKEND=pymc`` to cross-check), but it compiles once
and is reused across every Monte Carlo fit instead of rebuilding and recompiling a
PyMC graph per dataset.

Model (identical to ``methods._regression`` + ``methods._fit_uip``)
-------------------------------------------------------------------
Design D = [1, Z, Xc] with Xc = X - centroid, coefficient vector
phi = (beta0, theta, b_1..b_p).

    continuous : y_i ~ Normal((D phi)_i, sigma^2),  sigma ~ HalfNormal(2)
    binary     : y_i ~ Bernoulli(sigmoid((D phi)_i))

    theta ~ Normal(0, WEAK_SD^2),  b_j ~ Normal(0, 2.5^2)

    beta0 prior:
      no borrowing / pooling : Normal(0, WEAK_SD^2)
      UIP                    : Normal(mu(w), 1 / (M * s(w))),
                               mu(w) = sum_k w_k muC_k,  s(w) = sum_k w_k I_U,k,
                               w ~ Dirichlet(gamma),  M ~ Uniform(M_lo, M_hi)

Sampling
--------
* phi: exact Gaussian block update. Continuous uses the normal equations with
  precision D'D/sigma^2 + P0; binary uses Polya-Gamma augmentation
  (omega_i ~ PG(1, eta_i)) which makes the same update Gaussian.
* sigma (continuous): random-walk Metropolis on log sigma (HalfNormal(2) target).
* M (UIP): the full conditional is Gamma(3/2, rate = s * (beta0 - mu)^2 / 2)
  truncated to (M_lo, M_hi); sampled by random-walk Metropolis on log M.
* w (UIP, >=2 sources): Dirichlet random-walk Metropolis.
Step sizes for the three Metropolis moves adapt during warmup toward ~0.35
acceptance. Coefficients mix by exact Gibbs, so only the scalars need tuning.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from numba import njit

import config

PI = math.pi
TRUNC = 0.64


# --------------------------------------------------------------------------- #
# Polya-Gamma PG(1, z) sampler (Polson, Scott & Windle 2013, Devroye method).
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=True)
def _log_phi(x):
    # log standard-normal CDF via erfc, numerically fine over the used range.
    return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))


@njit(cache=True, fastmath=True)
def _mass_texpon(z):
    t = TRUNC
    fz = 0.125 * PI * PI + 0.5 * z * z
    b = math.sqrt(1.0 / t) * (t * z - 1.0)
    a = -math.sqrt(1.0 / t) * (t * z + 1.0)
    x0 = math.log(fz) + fz * t
    xb = x0 - z + _log_phi(b)
    xa = x0 + z + _log_phi(a)
    qdivp = 4.0 / PI * (math.exp(xb) + math.exp(xa))
    return 1.0 / (1.0 + qdivp)


@njit(cache=True, fastmath=True)
def _a_coef(n, x):
    k = (n + 0.5) * PI
    if x > TRUNC:
        return k * math.exp(-0.5 * k * k * x)
    elif x > 0.0:
        return k * (2.0 / (PI * x)) ** 1.5 * math.exp(-2.0 * (n + 0.5) ** 2 / x)
    return 0.0


@njit(cache=True, fastmath=True)
def _rtigauss(z):
    # Truncated (on (0, TRUNC)) inverse-Gaussian, mean 1/z, shape 1.
    z = abs(z)
    t = TRUNC
    x = t + 1.0
    if (1.0 / z) > t:
        alpha = 0.0
        while np.random.random() > alpha:
            e1 = np.random.exponential()
            e2 = np.random.exponential()
            while e1 * e1 > 2.0 * e2 / t:
                e1 = np.random.exponential()
                e2 = np.random.exponential()
            x = t / ((1.0 + t * e1) * (1.0 + t * e1))
            alpha = math.exp(-0.5 * z * z * x)
    else:
        mu = 1.0 / z
        while x > t:
            y = np.random.normal()
            y = y * y
            half_mu = 0.5 * mu
            mu_y = mu * y
            x = mu + half_mu * mu_y - half_mu * math.sqrt(4.0 * mu_y + mu_y * mu_y)
            if np.random.random() > mu / (mu + x):
                x = mu * mu / x
    return x


@njit(cache=True, fastmath=True)
def _pg1(zin):
    # One draw X ~ PG(1, zin).
    z = abs(zin) * 0.5
    fz = 0.125 * PI * PI + 0.5 * z * z
    while True:
        if np.random.random() < _mass_texpon(z):
            x = TRUNC + np.random.exponential() / fz
        else:
            x = _rtigauss(z)
        s = _a_coef(0, x)
        y = np.random.random() * s
        n = 0
        while True:
            n += 1
            if n % 2 == 1:
                s -= _a_coef(n, x)
                if y <= s:
                    return 0.25 * x
            else:
                s += _a_coef(n, x)
                if y > s:
                    break


# --------------------------------------------------------------------------- #
# Gibbs core.
# --------------------------------------------------------------------------- #
@njit(cache=True, fastmath=True)
def _gibbs(D, y, is_binary, has_uip, K,
           muC, I_U, gamma, M_lo, M_hi,
           beta0_mean0, beta0_prec0, weak_theta_var, b_var,
           n_draws, n_tune, seed):
    np.random.seed(seed)
    n = D.shape[0]
    q = D.shape[1]
    p_slope = q - 2

    DtD = D.T @ D
    Dty = D.T @ y
    kappa = y - 0.5  # binary working response

    # Diagonal prior precisions for theta and b (index 0 = beta0, set per iter).
    P0 = np.empty(q)
    P0[0] = 0.0
    P0[1] = 1.0 / weak_theta_var
    for j in range(2, q):
        P0[j] = 1.0 / b_var
    m0 = np.zeros(q)

    # Initialise phi by ridge-regularised least squares on the working response.
    yw = kappa if is_binary else y
    A0 = DtD.copy()
    for j in range(q):
        A0[j, j] += 1.0
    phi = np.linalg.solve(A0, D.T @ yw)
    resid = y - D @ phi
    sigma = max(np.sqrt(np.sum(resid * resid) / max(n - q, 1)), 1e-2)

    w = np.empty(K)
    gsum = 0.0
    for k in range(K):
        gsum += gamma[k]
    for k in range(K):
        w[k] = gamma[k] / gsum
    M = min(M_hi, max(M_lo, 0.5 * (M_lo + M_hi)))

    # Adaptive RW step sizes. NSWEEP inner Metropolis sweeps per outer draw for
    # the cheap (w, M) scalars so they mix as well as NUTS.
    NSWEEP = 8
    step_sig = 0.2
    step_M = 0.5
    conc_w = 200.0
    acc_sig = 0.0
    acc_M = 0.0
    acc_w = 0.0
    tot = 0.0

    theta_out = np.empty(n_draws)
    M_out = np.empty(n_draws)
    w_out = np.zeros(K)
    omega = np.empty(n)

    total = n_tune + n_draws
    for it in range(total):
        # ---- beta0 prior from (w, M) ----
        if has_uip:
            sW = 0.0
            muW = 0.0
            for k in range(K):
                sW += w[k] * I_U[k]
                muW += w[k] * muC[k]
            P0[0] = M * sW
            m0[0] = muW
        else:
            P0[0] = beta0_prec0
            m0[0] = beta0_mean0

        # ---- phi update (Gaussian block) ----
        if is_binary:
            eta = D @ phi
            for i in range(n):
                # Clamp the linear predictor before the PG draw and floor omega.
                # |eta| > 30 is an already-saturated logit; without this guard a
                # stray large draw makes omega ~ 0, Lam near-singular, and the
                # Gaussian phi update runs away (rare divergent chains at small n).
                e = eta[i]
                if e > 30.0:
                    e = 30.0
                elif e < -30.0:
                    e = -30.0
                om = _pg1(e)
                if om < 1e-6:
                    om = 1e-6
                omega[i] = om
            Lam = (D.T * omega) @ D
            rhs = D.T @ kappa
        else:
            inv_s2 = 1.0 / (sigma * sigma)
            Lam = DtD * inv_s2
            rhs = Dty * inv_s2
        for j in range(q):
            Lam[j, j] += P0[j]
            rhs[j] += P0[j] * m0[j]
        L = np.linalg.cholesky(Lam)
        mean = np.linalg.solve(Lam, rhs)
        z = np.random.standard_normal(q)
        noise = np.linalg.solve(np.ascontiguousarray(L.T), z)
        phi = mean + noise

        # ---- sigma update (continuous), MH on log sigma ----
        if not is_binary:
            r = y - D @ phi
            ssr = np.sum(r * r)
            ls = math.log(sigma)
            lp = -n * ls - 0.5 * ssr / (sigma * sigma) - 0.125 * sigma * sigma + ls
            ls_p = ls + step_sig * np.random.normal()
            sp = math.exp(ls_p)
            lp_p = -n * ls_p - 0.5 * ssr / (sp * sp) - 0.125 * sp * sp + ls_p
            if math.log(np.random.random() + 1e-300) < lp_p - lp:
                sigma = sp
                if it < n_tune:
                    acc_sig += 1.0

        # ---- M and w updates (UIP only). These scalars are cheap (no Cholesky /
        # no PG), so several inner Metropolis sweeps per outer draw sharpen their
        # mixing to match NUTS at negligible cost. ----
        b0 = phi[0]
        nsw = NSWEEP if has_uip else 0
        for _sweep in range(nsw):
            sW = 0.0
            muW = 0.0
            for k in range(K):
                sW += w[k] * I_U[k]
                muW += w[k] * muC[k]
            d2 = (b0 - muW) * (b0 - muW)
            # M: target log f(M) = 1.5*logM - 0.5*M*sW*d2 (jacobian incl.), in (M_lo,M_hi)
            lM = math.log(M)
            lM_p = lM + step_M * np.random.normal()
            Mp = math.exp(lM_p)
            if M_lo < Mp < M_hi:
                cur = 1.5 * lM - 0.5 * M * sW * d2
                prop = 1.5 * lM_p - 0.5 * Mp * sW * d2
                if math.log(np.random.random() + 1e-300) < prop - cur:
                    M = Mp
                    if it < n_tune:
                        acc_M += 1.0

            # w: Dirichlet RW proposal, target Dir(gamma) * N(beta0 | muW, 1/(M sW))
            if K >= 2:
                a_prop = np.empty(K)
                for k in range(K):
                    a_prop[k] = conc_w * w[k] + 1e-6
                wp = np.empty(K)
                gtot = 0.0
                for k in range(K):
                    wp[k] = np.random.gamma(a_prop[k], 1.0)
                    gtot += wp[k]
                for k in range(K):
                    wp[k] = wp[k] / gtot
                sWp = 0.0
                muWp = 0.0
                for k in range(K):
                    sWp += wp[k] * I_U[k]
                    muWp += wp[k] * muC[k]
                # log target (up to const): Dirichlet + Normal(beta0)
                lt_cur = 0.0
                lt_prop = 0.0
                for k in range(K):
                    lt_cur += (gamma[k] - 1.0) * math.log(w[k] + 1e-300)
                    lt_prop += (gamma[k] - 1.0) * math.log(wp[k] + 1e-300)
                precc = M * sW
                precp = M * sWp
                lt_cur += 0.5 * math.log(precc) - 0.5 * precc * (b0 - muW) ** 2
                lt_prop += 0.5 * math.log(precp) - 0.5 * precp * (b0 - muWp) ** 2
                # proposal correction q(w|wp)/q(wp|w), Dirichlet densities
                a_back = np.empty(K)
                for k in range(K):
                    a_back[k] = conc_w * wp[k] + 1e-6
                lq_fwd = 0.0
                lq_bwd = 0.0
                sfa = 0.0
                sba = 0.0
                for k in range(K):
                    sfa += a_prop[k]
                    sba += a_back[k]
                lq_fwd += math.lgamma(sfa)
                lq_bwd += math.lgamma(sba)
                for k in range(K):
                    lq_fwd += -math.lgamma(a_prop[k]) + (a_prop[k] - 1.0) * math.log(wp[k] + 1e-300)
                    lq_bwd += -math.lgamma(a_back[k]) + (a_back[k] - 1.0) * math.log(w[k] + 1e-300)
                if math.log(np.random.random() + 1e-300) < (lt_prop - lt_cur) + (lq_bwd - lq_fwd):
                    for k in range(K):
                        w[k] = wp[k]
                    if it < n_tune:
                        acc_w += 1.0

        # ---- adapt during warmup ----
        if it < n_tune:
            tot += 1.0
            if (it + 1) % 50 == 0 and tot > 0:
                if not is_binary:
                    ar = acc_sig / tot
                    step_sig *= 1.15 if ar > 0.4 else (0.85 if ar < 0.25 else 1.0)
                if has_uip:
                    denom = tot * NSWEEP
                    arM = acc_M / denom
                    step_M *= 1.15 if arM > 0.4 else (0.85 if arM < 0.25 else 1.0)
                    if K >= 2:
                        arw = acc_w / denom
                        # higher concentration -> smaller step
                        conc_w = min(8000.0, conc_w * 0.8) if arw > 0.4 else (
                            max(20.0, conc_w * 1.25) if arw < 0.2 else conc_w)
                acc_sig = 0.0
                acc_M = 0.0
                acc_w = 0.0
                tot = 0.0
        else:
            i_store = it - n_tune
            theta_out[i_store] = phi[1]
            M_out[i_store] = M
            for k in range(K):
                w_out[k] += w[k]

    if n_draws > 0:
        for k in range(K):
            w_out[k] /= n_draws
    return theta_out, M_out, w_out


# --------------------------------------------------------------------------- #
# Python wrappers returning the same summary dict shape as methods._summary.
# --------------------------------------------------------------------------- #
def _design(X, Z, center):
    Xc = np.asarray(X, float) - center
    n = Xc.shape[0]
    D = np.empty((n, Xc.shape[1] + 2))
    D[:, 0] = 1.0
    D[:, 1] = np.asarray(Z, float)
    D[:, 2:] = Xc
    return np.ascontiguousarray(D)


def _collect(theta, Mdraws, w_mean, K, decision_cut=config.DECISION_CUT):
    th = np.asarray(theta)
    ps = float(np.mean(th > 0.0))
    out = {"theta_mean": float(th.mean()), "theta_sd": float(th.std(ddof=1)),
           "ci_lo": float(np.quantile(th, 0.025)),
           "ci_hi": float(np.quantile(th, 0.975)),
           "prob_sup": ps, "decision": int(ps > decision_cut),
           "M_mean": float(np.mean(Mdraws)), "rhat": np.nan}
    return out


def run_fixed(outcome, X, Z, Y, center, beta0_var, weak_theta_var, mcmc, seed):
    """no_borrowing / pooling: fixed weak Normal(0, beta0_var) prior on beta0."""
    D = _design(X, Z, center)
    y = np.ascontiguousarray(np.asarray(Y, float))
    is_bin = outcome != "continuous"
    dummy = np.ones(1)
    theta, Md, wm = _gibbs(
        D, y, is_bin, False, 1, dummy, dummy, dummy, 1e-3, 1.0,
        0.0, 1.0 / beta0_var, weak_theta_var, 2.5 * 2.5,
        int(mcmc["draws"]) * int(mcmc.get("chains", 2)),
        int(mcmc["tune"]), int(seed))
    return _collect(theta, Md, wm, 1)


def run_uip_fixed(outcome, X, Z, Y, center, mu_prior, prec_prior,
                  weak_theta_var, mcmc, seed):
    """Deterministic UIP borrowing: a fixed informative Normal(mu_prior,
    1/prec_prior) prior on beta0, with the borrowed control mean and precision
    computed at the design stage (standard_uip, ps_uip_c, ps_power_prior). No
    sampled M, no Dirichlet weights -- the borrowing amount is settled before
    sampling, so this reuses the no-borrowing Gibbs path with an informative beta0
    prior. M_mean in the returned dict is a placeholder; the caller overwrites it
    with the deterministic borrowed sample size."""
    D = _design(X, Z, center)
    y = np.ascontiguousarray(np.asarray(Y, float))
    is_bin = outcome != "continuous"
    dummy = np.ones(1)
    theta, Md, wm = _gibbs(
        D, y, is_bin, False, 1, dummy, dummy, dummy, 1e-3, 1.0,
        float(mu_prior), float(prec_prior), weak_theta_var, 2.5 * 2.5,
        int(mcmc["draws"]) * int(mcmc.get("chains", 2)),
        int(mcmc["tune"]), int(seed))
    return _collect(theta, Md, wm, 1)


def run_uip(outcome, X, Z, Y, center, muC, I_U, gamma, M_lo, M_hi,
            weak_theta_var, mcmc, seed):
    """UIP borrowing on beta0 with sampled (w, M)."""
    D = _design(X, Z, center)
    y = np.ascontiguousarray(np.asarray(Y, float))
    is_bin = outcome != "continuous"
    K = len(muC)
    theta, Md, wm = _gibbs(
        D, y, is_bin, True, K,
        np.ascontiguousarray(np.asarray(muC, float)),
        np.ascontiguousarray(np.asarray(I_U, float)),
        np.ascontiguousarray(np.asarray(gamma, float)),
        float(M_lo), float(M_hi), 0.0, 1.0, weak_theta_var, 2.5 * 2.5,
        int(mcmc["draws"]) * int(mcmc.get("chains", 2)),
        int(mcmc["tune"]), int(seed))
    return _collect(theta, Md, wm, K), wm
