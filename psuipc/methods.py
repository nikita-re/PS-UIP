"""
psuipc/methods.py
=================

Stage 2 of PS-UIP and its benchmarks. The current two-arm RCT is fit with the
single regression used by Zhang & Yin (2023), identical in form to the
data-generating model,

    g(E[Y]) = beta0 + theta * Z + X^T b ,

with g the identity link (continuous) or the logit link (binary). Covariates are
centered at the current control centroid, so beta0 is the control outcome at the
centroid, the control parameter informed by the historical sources. The coefficient
theta is the common conditional mean difference for a continuous endpoint and the
common conditional log odds ratio for a binary endpoint under the analysis model.
Historical sources are control-only and provide no direct treatment comparison or
direct likelihood information about theta. The unit information prior is placed on
beta0, while theta retains a weak prior. Borrowing for beta0 can affect the posterior
of theta through the joint current-trial regression model.

Decision: Pr(theta > 0 | data) > 0.95  (one-sided 5%).

Methods
-------
no_borrowing   : current RCT only, vague control prior.
pooling        : stack current + all raw historical controls in the control
                 likelihood (naive, no adjustment).
standard_uip   : fixed full-information UIP variant, motivated by Zhang & Yin
                 (2023) / Jin & Yin (2021) and Kass & Wasserman (1995). It borrows
                 all n_k historical control units, with no propensity-score,
                 overlap, or conflict adjustment.
ps_power_prior : matched PS-weighted power-prior variant motivated by Lu (2022)
                 and Li (2025). It uses transported control summaries and a fixed
                 overlap-budgeted power exponent but no outcome-conflict discount.
ps_uip_c       : PROPOSED PS-UIP. PS-balanced control summaries (Stage 1: membership
                 PS + IPTW, giving the Kish ESS m_k and overlap r_k) discounted by a
                 prior-data-conflict factor rho_k* (Stage 2), the stronger of a
                 per-source factor and an aggregate factor over the combined sources.
                 Information units M_k = m_k r_k rho_k* <= n_k feed a deterministic
                 unit information prior. The bound limits borrowing but does not
                 guarantee type-I error control.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, List, Optional

import numpy as np

import config
import psuipc.stage1 as s1
import psuipc.fast_sampler as fs

# Sampling backend: "c" (compiled C Gibbs, default), "fast" (Numba Gibbs), or
# "pymc" (NUTS via nutpie/PyMC).
BACKEND = os.environ.get("PSUIPC_BACKEND", "c").lower()

# PyMC and ArviZ are imported only when that optional backend is requested. This
# keeps the default C/Numba path independent of optional packages and their runtime
# configuration.
pm = az = None
if BACKEND == "pymc":
    try:
        import pymc as pm
        import arviz as az
    except ImportError:  # pragma: no cover
        pm = az = None

DECISION_CUT = config.DECISION_CUT
WEAK_SD = config.WEAK_THETA_SD

# All three backends target the identical posterior, so they
# are statistically interchangeable; cross-check with PSUIPC_BACKEND=pymc. The "c"
# and "fast" backends both expose run_fixed / run_uip_fixed / run_uip; _be selects
# which module the Gibbs-path calls below go through. The "pymc" path (BACKEND not
# in {fast, c}) is unchanged.
# The "c" backend is the ctypes wrapper around the compiled C Gibbs sampler
# (psuipc/csrc/sampler.c -> psuipc/_csampler.dll). It is the default because it has
# no JIT warmup. If the shared library is missing or cannot be loaded (the .dll is a
# gitignored build artifact), fall back to the numba "fast" backend so a fresh
# checkout still runs; build the library with psuipc/csrc/build.sh.
if BACKEND == "c":
    try:
        import psuipc.c_backend as cb
        _be = cb
    except Exception as _e:  # pragma: no cover
        warnings.warn(f"PSUIPC_BACKEND=c but the C sampler could not be loaded "
                      f"({type(_e).__name__}: {_e}); falling back to the numba 'fast' "
                      f"backend. Build the library with psuipc/csrc/build.sh.")
        BACKEND = "fast"
        _be = fs
else:
    _be = fs

# True when a Gibbs backend (compiled "c" or numba "fast") handles sampling.
_GIBBS = BACKEND in ("fast", "c")


def _sample(model, mcmc, seed):
    if pm is None:
        raise ImportError("the 'pymc' backend requires pymc and arviz; install them "
                          "or use the Gibbs backend (PSUIPC_BACKEND=fast)")
    common = dict(draws=mcmc["draws"], tune=mcmc["tune"], chains=mcmc["chains"],
                  cores=1, target_accept=mcmc.get("target_accept", 0.9),
                  random_seed=seed, progressbar=False,
                  compute_convergence_checks=False)
    with model:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idata = pm.sample(nuts_sampler="nutpie", **common)
            return idata, "nutpie"
        except Exception:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idata = pm.sample(nuts_sampler="pymc", **common)
            return idata, "pymc"


def _summary(idata):
    th = np.asarray(idata.posterior["theta"].values).reshape(-1)
    ps = float(np.mean(th > 0.0))
    out = {"theta_mean": float(th.mean()), "theta_sd": float(th.std(ddof=1)),
           "ci_lo": float(np.quantile(th, 0.025)), "ci_hi": float(np.quantile(th, 0.975)),
           "prob_sup": ps, "decision": int(ps > DECISION_CUT)}
    try:
        out["rhat"] = float(az.rhat(idata, var_names=["theta"])["theta"].values)
    except Exception:
        out["rhat"] = np.nan
    return out


def _center(cur):
    """Current control-arm covariate centroid; the borrowing scale reference."""
    return np.asarray(cur["X_C"]).mean(0)


def _regression(outcome, beta0, X, Z, Y, center):
    """Single covariate-adjusted regression g(E[Y]) = beta0 + theta*Z + Xc^T b,
    Xc = X - center. beta0 (the control intercept, carrying any borrowed prior)
    is supplied by the caller; theta (weak prior) and the covariate slope b are
    local. Stacking historical CONTROL rows (Z=0) into (X,Z,Y) implements pooling
    within the very same regression."""
    Xc = np.asarray(X) - center
    Zv = np.asarray(Z, float)
    theta = pm.Normal("theta", 0.0, WEAK_SD)
    b = pm.Normal("b", 0.0, 2.5, shape=Xc.shape[1])
    eta = beta0 + theta * Zv + pm.math.dot(Xc, b)
    if outcome == "continuous":
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=eta, sigma=sigma, observed=np.asarray(Y))
    else:
        pm.Bernoulli("y", logit_p=eta, observed=np.asarray(Y))


def _blank(method):
    d = {"method": method, "theta_mean": np.nan, "theta_sd": np.nan,
         "ci_lo": np.nan, "ci_hi": np.nan, "prob_sup": np.nan, "decision": np.nan,
         "rhat": np.nan, "M_mean": np.nan, "sampler": "", "failed": 0, "reason": ""}
    for k in range(1, 4):  # K = 3 sources
        d[f"w{k}_mean"] = np.nan
        d[f"m{k}"] = np.nan
        d[f"r{k}"] = np.nan
        d[f"rho{k}"] = np.nan
    return d


# ---------------------------------------------------------------------------
def fit_no_borrowing(outcome, cur, sources, mcmc, seed):
    res = _blank("no_borrowing")
    try:
        center = _center(cur)
        if _GIBBS:
            out = _be.run_fixed(outcome, cur["X"], cur["Z"], cur["Y"], center,
                                WEAK_SD ** 2, WEAK_SD ** 2, mcmc, seed)
            res.update(out); res["sampler"] = BACKEND; res["M_mean"] = 0.0
            return res
        with pm.Model() as model:
            beta0 = pm.Normal("beta0", 0.0, WEAK_SD)
            _regression(outcome, beta0, cur["X"], cur["Z"], cur["Y"], center)
        idata, samp = _sample(model, mcmc, seed)
        res.update(_summary(idata)); res["sampler"] = samp; res["M_mean"] = 0.0
    except Exception as e:
        res["failed"] = 1; res["reason"] = f"{type(e).__name__}: {e}"
    return res


def fit_pooling(outcome, cur, sources, mcmc, seed):
    res = _blank("pooling")
    try:
        center = _center(cur)  # reference fixed at the current control centroid
        # Stack every historical control as a Z=0 row in the same regression.
        Xs = np.vstack([np.asarray(cur["X"])] + [np.asarray(s["X"]) for s in sources])
        Zs = np.concatenate([np.asarray(cur["Z"], float)]
                            + [np.zeros(len(s["Y"])) for s in sources])
        Ys = np.concatenate([np.asarray(cur["Y"])] + [np.asarray(s["Y"]) for s in sources])
        if _GIBBS:
            out = _be.run_fixed(outcome, Xs, Zs, Ys, center,
                                WEAK_SD ** 2, WEAK_SD ** 2, mcmc, seed)
            res.update(out); res["sampler"] = BACKEND
            res["M_mean"] = float(sum(len(s["Y"]) for s in sources))
            return res
        with pm.Model() as model:
            beta0 = pm.Normal("beta0", 0.0, WEAK_SD)
            _regression(outcome, beta0, Xs, Zs, Ys, center)
        idata, samp = _sample(model, mcmc, seed)
        res.update(_summary(idata)); res["sampler"] = samp
        res["M_mean"] = float(sum(len(s["Y"]) for s in sources))
    except Exception as e:
        res["failed"] = 1; res["reason"] = f"{type(e).__name__}: {e}"
    return res


# ---------------------------------------------------------------------------
# Deterministic unit information prior (standard_uip, ps_power_prior, ps_uip_c).
# The borrowed control mean and precision are computed at the design stage, so a
# fixed informative Normal prior on beta0 is enough -- no sampled (w, M).
# ---------------------------------------------------------------------------
def _uip_prior(summaries, units):
    """Unit information prior on beta0 (Zhang & Yin 2023; Jin & Yin 2021). Given valid
    per-source summaries [{muC,SE,m,...}] and the effective borrowed unit counts
    ``units``, return (mu_prior, prec_prior, M_det, w). Each source contributes
    units_k * I_{U,k} = units_k / (m_k SE_k^2) to the prior precision, where
    I_{U,k} = 1/(m_k SE_k^2) is the Fisher information of ONE source-k observation;
    mu_prior is the precision-weighted mean of the source control means and
    M_det = sum_k units_k is the borrowed effective sample size, the UIP parameter."""
    contrib = np.array([u / (s["m"] * s["SE"] ** 2)
                        for u, s in zip(units, summaries)], float)
    prec = float(contrib.sum())
    if not np.isfinite(prec) or prec <= 0:
        return 0.0, 0.0, 0.0, np.zeros(len(summaries))
    muC = np.array([s["muC"] for s in summaries], float)
    mu = float(np.sum(contrib * muC) / prec)
    return mu, prec, float(np.sum(units)), contrib / prec


def _fixed_uip_fit(outcome, cur, summaries, units, mcmc, seed, method, rho=None):
    """Fit the current RCT with a deterministic informative beta0 prior built from
    per-source summaries and borrowed unit counts. Records the per-source design
    diagnostics (m_k, r_k, rho_k, w_k). Falls back to no borrowing when nothing is
    borrowable."""
    res = _blank(method)
    for i, s in enumerate(summaries):
        if s is not None:
            res[f"m{i+1}"] = s["m"]; res[f"r{i+1}"] = s["r"]
        if rho is not None:
            res[f"rho{i+1}"] = float(rho[i])
    idx = [i for i, s in enumerate(summaries) if s is not None]
    valid = [summaries[i] for i in idx]
    uvalid = [max(0.0, float(units[i])) for i in idx]
    if len(valid) == 0 or sum(uvalid) <= 1e-8:
        base = fit_no_borrowing(outcome, cur, None, mcmc, seed)
        res.update({k: base[k] for k in base if k not in
                    ("method",) and not k.startswith(("m", "r", "w", "rho"))})
        res["method"] = method; res["M_mean"] = 0.0
        return res
    mu_prior, prec_prior, M_det, w = _uip_prior(valid, uvalid)
    try:
        center = _center(cur)
        if _GIBBS:
            out = _be.run_uip_fixed(outcome, cur["X"], cur["Z"], cur["Y"], center,
                                    mu_prior, prec_prior, WEAK_SD ** 2, mcmc, seed)
            res.update(out); res["sampler"] = BACKEND
        else:
            sd_prior = 1.0 / math.sqrt(prec_prior)
            with pm.Model() as model:
                beta0 = pm.Normal("beta0", mu=mu_prior, sigma=sd_prior)
                _regression(outcome, beta0, cur["X"], cur["Z"], cur["Y"], center)
            idata, samp = _sample(model, mcmc, seed)
            res.update(_summary(idata)); res["sampler"] = samp
        res["M_mean"] = float(M_det)
        for j, i in enumerate(idx):
            res[f"w{i+1}_mean"] = float(w[j])
    except Exception as e:
        res["failed"] = 1; res["reason"] = f"{type(e).__name__}: {e}"
    return res


def fit_standard_uip(outcome, cur, sources, mcmc, seed):
    """Fixed full-information UIP variant motivated by Zhang & Yin (2023), Jin &
    Yin (2021), and Kass & Wasserman (1995). It borrows all n_k historical control
    units with no PS, overlap, or conflict adjustment. For raw summaries m_k = n_k,
    so prior precision reduces to sum_k 1/SE_k^2. This benchmark is not a verbatim
    implementation of the adaptive weight models in the cited UIP articles."""
    summaries = [s1.raw_control_summary(outcome, cur["X_C"], s["X"], s["Y"])
                 for s in sources]
    units = [s["m"] if s is not None else 0.0 for s in summaries]
    return _fixed_uip_fit(outcome, cur, summaries, units, mcmc, seed, "standard_uip")


def fit_ps_power_prior(outcome, cur, sources, mcmc, seed):
    """Matched PS-weighted power-prior variant motivated by Lu (2022) and Li (2025): transported control summaries
    with a fixed overlap-budgeted borrow. Borrowed units
    a_k = min(A r_k / sum_j r_j, m_k), A = config.POWER_PRIOR_A (defaults to the
    current control size nC). It has the propensity score but no conflict discount."""
    summaries = [s1.control_summary(outcome, cur["X_C"], s["X"], s["Y"])
                 for s in sources]
    A = config.POWER_PRIOR_A if config.POWER_PRIOR_A is not None else float(cur["nC"])
    rsum = sum(s["r"] for s in summaries if s is not None)
    units = []
    for s in summaries:
        if s is None or rsum <= 0:
            units.append(0.0)
        else:
            units.append(min(A * s["r"] / rsum, s["m"]))
    return _fixed_uip_fit(outcome, cur, summaries, units, mcmc, seed, "ps_power_prior")


def _aggregate_discount(summaries, base_units, ref, c=None, shape="exp"):
    """Aggregate prior-data-conflict factor rho_g for the OVERLAP-WEIGHTED COMBINED
    control summary. A drift shared across sources can leave each per-source
    discrepancy T_k below one chi-square unit (so rho_k = 1) yet move the combined
    borrowed mean well away from the current control, because the combined summary
    has a much smaller standard error (it pools the sources). Forming the combined
    (mu_pool, SE_pool) from the inverse-variance, overlap-weighted sources and
    applying the same exponential discount as conflict_discount() catches that common
    drift. The per-source factor catches a single deviant source; the aggregate
    factor catches a coordinated one; the borrowing uses whichever is stronger."""
    if ref is None:
        return 1.0
    c = config.RHO_C if c is None else c
    idx = [i for i, s in enumerate(summaries) if s is not None and base_units[i] > 0]
    if not idx:
        return 1.0
    mu_pool, prec_pool, _, _ = _uip_prior([summaries[i] for i in idx],
                                          [base_units[i] for i in idx])
    se_pool2 = 1.0 / max(prec_pool, 1e-12)
    T = (mu_pool - ref["muC"]) ** 2 / (se_pool2 + ref["SE"] ** 2)
    if not np.isfinite(T) or T <= c:
        return 1.0
    return float(c / T) if shape == "invT" else float(math.exp(c - T))


def fit_ps_uip_c(outcome, cur, sources, mcmc, seed, c=None, shape="exp"):
    """PROPOSED PS-UIP. Stage 1 gives PS-balanced summaries with Kish ESS m_k and
    overlap r_k; Stage 2 discounts each source by the minimum of a per-source conflict
    factor rho_k (its own control mean vs the current control) and an aggregate factor
    rho_g (the overlap-weighted combined control mean vs the current control), both
    the exponential discount of conflict_discount(). The per-source factor stops a
    single deviant source; the aggregate factor stops a drift shared across sources
    that each one alone hides, the case that matters for low-information binary
    endpoints. Information units M_k = m_k r_k min(rho_k, rho_g) <= n_k feed a
    deterministic unit information prior on beta0. The bound limits borrowing but
    does not guarantee type-I error control."""
    c = config.RHO_C if c is None else c
    summaries = [s1.control_summary(outcome, cur["X_C"], s["X"], s["Y"])
                 for s in sources]
    ref = s1.current_control_summary(outcome, cur["X_C"], cur["Y_C"])
    base = [(s["m"] * s["r"] if s is not None else 0.0) for s in summaries]
    rho_g = _aggregate_discount(summaries, base, ref, c, shape)
    rho, units = [], []
    for s in summaries:
        if s is None or ref is None:
            rho.append(1.0); units.append(0.0)
        else:
            rk = s1.conflict_discount(s["muC"], s["SE"], ref["muC"], ref["SE"],
                                      c, shape)
            r = min(rk, rho_g)
            rho.append(r); units.append(s["m"] * s["r"] * r)
    return _fixed_uip_fit(outcome, cur, summaries, units, mcmc, seed, "ps_uip_c",
                          rho=rho)


def fit_ps_uip_psonly(outcome, cur, sources, mcmc, seed):
    """ABLATION of PS-UIP that drops the aggregate factor rho_g and discounts each
    source by its per-source factor rho_k alone, M_k = m_k r_k rho_k. Comparing it
    with ps_uip_c isolates the marginal value of rho_g (the contrast that matters for
    a drift shared across sources, which each per-source test alone can miss), and the
    gap in the COMPATIBLE sources' borrowed units between the two is the cost of the
    min-combine when one source is in conflict (it over-discounts the innocent ones)."""
    summaries = [s1.control_summary(outcome, cur["X_C"], s["X"], s["Y"])
                 for s in sources]
    ref = s1.current_control_summary(outcome, cur["X_C"], cur["Y_C"])
    rho, units = [], []
    for s in summaries:
        if s is None or ref is None:
            rho.append(1.0); units.append(0.0)
        else:
            rk = s1.conflict_discount(s["muC"], s["SE"], ref["muC"], ref["SE"],
                                      config.RHO_C)
            rho.append(rk); units.append(s["m"] * s["r"] * rk)
    return _fixed_uip_fit(outcome, cur, summaries, units, mcmc, seed, "ps_uip_psonly",
                          rho=rho)


def _theta_fit(outcome, cur, mu_prior, prec_prior, mcmc, seed):
    """Fit the current RCT with a fixed informative Normal(mu_prior, 1/prec_prior)
    prior on beta0 and return (theta_mean, theta_sd). Used to build the two components
    of the PS-SAM mixture posterior."""
    center = _center(cur)
    if _GIBBS:
        out = _be.run_uip_fixed(outcome, cur["X"], cur["Z"], cur["Y"], center,
                                mu_prior, prec_prior, WEAK_SD ** 2, mcmc, seed)
        return float(out["theta_mean"]), float(out["theta_sd"])
    sd = 1.0 / math.sqrt(prec_prior)
    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=mu_prior, sigma=sd)
        _regression(outcome, beta0, cur["X"], cur["Z"], cur["Y"], center)
    idata, _ = _sample(model, mcmc, seed)
    th = np.asarray(idata.posterior["theta"].values).reshape(-1)
    return float(th.mean()), float(th.std(ddof=1))


def _sam_weight(mu_inf, prec_inf, ref):
    """Self-adapting mixture weight w in [0,1] on the informative component (Schmidli
    et al. 2014; Yang et al. 2023). It is the marginal-likelihood weight of the
    informative prior in a two-component mixture {informative, vague} for the current
    control summary, with equal prior odds, w = m_I / (m_I + m_V), where m_I and m_V are
    the marginal densities of the current control summary muC_cur under the informative
    prior N(mu_inf, 1/prec_inf) and a vague prior N(mu_inf, SAM_VAGUE_SD^2). Under
    agreement w -> 1; under a large control-mean conflict w -> 0."""
    if ref is None or not np.isfinite(prec_inf) or prec_inf <= 0:
        return 1.0
    var_inf = 1.0 / prec_inf + ref["SE"] ** 2
    var_vag = config.SAM_VAGUE_SD ** 2 + ref["SE"] ** 2
    d2 = (ref["muC"] - mu_inf) ** 2
    log_mI = -0.5 * math.log(var_inf) - 0.5 * d2 / var_inf
    log_mV = -0.5 * math.log(var_vag) - 0.5 * d2 / var_vag
    mx = max(log_mI, log_mV)
    mI, mV = math.exp(log_mI - mx), math.exp(log_mV - mx)
    return float(mI / (mI + mV)) if (mI + mV) > 0 else 1.0


def _mixture_theta(mI, sI, mV, sV, W):
    """Summarize the two-component Gaussian mixture posterior of theta,
    W * N(mI, sI^2) + (1 - W) * N(mV, sV^2): posterior mean, sd, central 95% interval
    (by bisection on the mixture CDF), superiority probability Pr(theta > 0), and the
    one-sided decision. The theta posterior of each fixed-prior fit is approximately
    Gaussian, so the mixture-prior posterior is the corresponding Gaussian mixture."""
    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    prob = W * Phi(mI / sI) + (1.0 - W) * Phi(mV / sV)
    mean = W * mI + (1.0 - W) * mV
    var = W * (sI ** 2 + mI ** 2) + (1.0 - W) * (sV ** 2 + mV ** 2) - mean ** 2
    sd = math.sqrt(max(var, 1e-12))

    def F(t):
        return W * Phi((t - mI) / sI) + (1.0 - W) * Phi((t - mV) / sV)

    def quant(p):
        lo = min(mI - 8 * sI, mV - 8 * sV)
        hi = max(mI + 8 * sI, mV + 8 * sV)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if F(mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return {"theta_mean": mean, "theta_sd": sd,
            "ci_lo": quant(0.025), "ci_hi": quant(0.975),
            "prob_sup": prob, "decision": int(prob > DECISION_CUT), "rhat": np.nan}


def fit_ps_sam(outcome, cur, sources, mcmc, seed):
    """PROPENSITY-SCORE SELF-ADAPTING MIXTURE PRIOR benchmark (Schmidli et al. 2014;
    Yang et al. 2023; Zhao et al. 2025), the closest published dynamic competitor to
    PS-UIP. It uses the same PS-balanced control summaries as PS-UIP but replaces the
    deterministic conflict discount with a genuine two-component robust mixture prior on
    beta0, w * N(mu_inf, 1/prec_inf) + (1 - w) * N(mu_inf, SAM_VAGUE_SD^2), where
    (mu_inf, prec_inf) is the PS-balanced unit information component and w is the
    self-adapting weight of _sam_weight(). The posterior of theta is the mixture of the
    two component-conditional posteriors, weighted by W = w m_I / (w m_I + (1-w) m_V);
    the vague component gives the robustness (heavy tail) that a mixture prior keeps and
    an effective-sample-size scaling would drop. The contrast with ps_uip_c isolates the
    conflict-response rule, deterministic discount rho_k* versus adaptive mixture weight,
    on the same PS-balanced information."""
    res = _blank("ps_sam")
    summaries = [s1.control_summary(outcome, cur["X_C"], s["X"], s["Y"])
                 for s in sources]
    for i, s in enumerate(summaries):
        if s is not None:
            res[f"m{i+1}"] = s["m"]; res[f"r{i+1}"] = s["r"]
    ref = s1.current_control_summary(outcome, cur["X_C"], cur["Y_C"])
    base = [(s["m"] * s["r"] if s is not None else 0.0) for s in summaries]
    idx = [i for i, s in enumerate(summaries) if s is not None and base[i] > 0]
    if not idx or ref is None:
        base_fit = fit_no_borrowing(outcome, cur, None, mcmc, seed)
        res.update({k: base_fit[k] for k in base_fit if not k.startswith(
            ("m", "r", "w", "rho")) and k != "method"})
        res["method"] = "ps_sam"; res["M_mean"] = 0.0
        return res
    try:
        mu_inf, prec_inf, M_inf, _ = _uip_prior([summaries[i] for i in idx],
                                                [base[i] for i in idx])
        var_inf = 1.0 / prec_inf + ref["SE"] ** 2
        var_vag = config.SAM_VAGUE_SD ** 2 + ref["SE"] ** 2
        d2 = (ref["muC"] - mu_inf) ** 2
        log_mI = -0.5 * math.log(var_inf) - 0.5 * d2 / var_inf
        log_mV = -0.5 * math.log(var_vag) - 0.5 * d2 / var_vag
        mx = max(log_mI, log_mV)
        mI_d, mV_d = math.exp(log_mI - mx), math.exp(log_mV - mx)
        w = mI_d / (mI_d + mV_d) if (mI_d + mV_d) > 0 else 1.0
        W = (w * mI_d) / (w * mI_d + (1.0 - w) * mV_d) if (w * mI_d + (1.0 - w) * mV_d) > 0 else 1.0
        prec_vag = 1.0 / (config.SAM_VAGUE_SD ** 2)
        mI, sI = _theta_fit(outcome, cur, mu_inf, prec_inf, mcmc, seed)
        mV, sV = _theta_fit(outcome, cur, mu_inf, prec_vag, mcmc, seed + 1)
        res.update(_mixture_theta(mI, sI, mV, sV, W))
        res["sampler"] = BACKEND
        res["M_mean"] = float(W * M_inf)  # effective borrowed ESS = posterior weight x informative ESS
        for i in idx:
            res[f"rho{i+1}"] = float(W)
    except Exception as e:  # pragma: no cover
        res["failed"] = 1; res["reason"] = f"{type(e).__name__}: {e}"
    return res


# ---------------------------------------------------------------------------
# Adaptive UIP with sampled (w, M) -- appendix robustness ablation only.
# ---------------------------------------------------------------------------
def _fit_uip(outcome, cur, summaries, mcmc, seed, method):
    """Shared fully-Bayesian control UIP given per-source summaries [{muC,SE,m,r}]."""
    res = _blank(method)
    valid = [s for s in summaries if s is not None]
    for i, s in enumerate(summaries):
        if s is not None:
            res[f"m{i+1}"] = s["m"]; res[f"r{i+1}"] = s["r"]
    if len(valid) == 0:
        return fit_no_borrowing(outcome, cur, None, mcmc, seed) | {"method": method}
    nC = cur["nC"]
    muC_vec = np.array([s["muC"] for s in valid])
    I_U = np.array([1.0 / (s["m"] * s["SE"] ** 2) for s in valid])
    mr = np.array([s["m"] * s["r"] for s in valid])
    gamma = np.maximum(np.minimum(1.0, mr / nC), 1e-3)
    M_cap = float(max(min(nC, mr.sum()), 1e-3 + 1e-6))
    try:
        center = _center(cur)
        if _GIBBS:
            out, wm = _be.run_uip(outcome, cur["X"], cur["Z"], cur["Y"], center,
                                  muC_vec, I_U, gamma, 1e-3, M_cap,
                                  WEAK_SD ** 2, mcmc, seed)
            res.update(out); res["sampler"] = BACKEND
            if len(valid) > 1:
                for i in range(len(valid)):
                    res[f"w{i+1}_mean"] = float(wm[i])
            else:
                res["w1_mean"] = 1.0
            return res
        with pm.Model() as model:
            if len(valid) == 1:
                w = pm.math.ones((1,))
            else:
                w = pm.Dirichlet("w", a=gamma)
            M = pm.Uniform("M", 1e-3, M_cap)
            mu_prior = pm.math.dot(w, muC_vec)
            prec = M * pm.math.dot(w, I_U)
            sd_prior = 1.0 / pm.math.sqrt(prec)
            beta0 = pm.Normal("beta0", mu=mu_prior, sigma=sd_prior)
            _regression(outcome, beta0, cur["X"], cur["Z"], cur["Y"], center)
        idata, samp = _sample(model, mcmc, seed)
        res.update(_summary(idata)); res["sampler"] = samp
        res["M_mean"] = float(np.mean(idata.posterior["M"].values))
        if len(valid) > 1:
            wp = idata.posterior["w"].values.reshape(-1, len(valid)).mean(0)
            for i in range(len(valid)):
                res[f"w{i+1}_mean"] = float(wp[i])
        else:
            res["w1_mean"] = 1.0
    except Exception as e:
        res["failed"] = 1; res["reason"] = f"{type(e).__name__}: {e}"
    return res


def fit_uip_noPS(outcome, cur, sources, mcmc, seed):
    """Appendix ablation: adaptive (sampled w, M) UIP on RAW no-PS summaries."""
    summaries = [s1.raw_control_summary(outcome, cur["X_C"], s["X"], s["Y"])
                 for s in sources]
    return _fit_uip(outcome, cur, summaries, mcmc, seed, "uip_noPS")


def fit_ps_uip_adaptive(outcome, cur, sources, mcmc, seed):
    """Appendix ablation: adaptive (sampled w, M) UIP on PS-balanced summaries,
    the pre-revision PS-UIP; kept to show the deterministic conflict discount is
    what earns type-I control, not the sampled-M self-discounting."""
    summaries = [s1.control_summary(outcome, cur["X_C"], s["X"], s["Y"])
                 for s in sources]
    return _fit_uip(outcome, cur, summaries, mcmc, seed, "ps_uip_adaptive")


# Main suite for the paper. Order is the display order.
METHODS = {
    "no_borrowing": fit_no_borrowing,
    "pooling": fit_pooling,
    "standard_uip": fit_standard_uip,
    "ps_power_prior": fit_ps_power_prior,
    "ps_uip_c": fit_ps_uip_c,
}

# Revision ablations / extra competitor (major-revision experiments #2 and #5):
# ps_uip_psonly drops the aggregate factor rho_g (isolates its value and the
# min-combine over-discount cost); ps_sam is the self-adapting mixture competitor.
METHODS_ABLATION = {
    "ps_uip_psonly": fit_ps_uip_psonly,
    "ps_sam": fit_ps_sam,
}

# Appendix ablations (adaptive sampled-(w, M) variants), not in the main suite.
METHODS_APPENDIX = {
    "uip_noPS": fit_uip_noPS,
    "ps_uip_adaptive": fit_ps_uip_adaptive,
}
