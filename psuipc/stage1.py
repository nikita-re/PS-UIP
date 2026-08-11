"""
psuipc/stage1.py
================

Stage 1 of PS-UIP (outcome-free, propensity-score design stage).

For each historical CONTROL source k, the membership propensity score
e_k(x) = Pr(current control | x) is fitted by pooling the current control arm
(G=1) against source-k controls (G=0). Historical controls are trimmed to the
current-control common support and transported to the current covariate
distribution by stabilized inverse-probability weighting. The PS-balanced source
then yields the control summary needed by the unit information prior:

    S_k = { muC_k, SE_k, m_k },   r_k

where muC_k is the PS-weighted control mean (continuous) or log-odds (binary),
SE_k its model-based standard error, m_k the Kish effective sample size after
weighting, and r_k the membership-PS overlap coefficient. This mirrors the
PS-integrated borrowing of Wang et al. (2019, 2022) and Zhao et al. (2025) but
produces a unit-information summary (Zhang & Yin, 2023) rather than a power-prior
exponent or a mixture weight.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

import config

PS_CLIP = config.PS_CLIP
MIN_RETAINED = config.MIN_RETAINED
MIN_ESS = config.MIN_ESS
N_BINS = config.N_OVERLAP_BINS
WINSOR = config.WINSOR_PCT


def kish_ess(omega) -> float:
    omega = np.asarray(omega, float)
    s2 = np.sum(omega ** 2)
    return float(omega.sum() ** 2 / s2) if s2 > 0 else 0.0


def membership_overlap(e_cur, e_hist, omega) -> float:
    """Overlap coefficient of the membership-PS densities (current controls vs
    omega-weighted historical controls)."""
    e_cur, e_hist, omega = np.asarray(e_cur), np.asarray(e_hist), np.asarray(omega, float)
    pooled = np.concatenate([e_cur, e_hist])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, N_BINS + 1)))
    if edges.size < 2:
        return 0.0
    edges[0] -= 1e-9; edges[-1] += 1e-9
    pc, _ = np.histogram(e_cur, bins=edges)
    ph, _ = np.histogram(e_hist, bins=edges, weights=omega)
    pc, ph = pc.astype(float), ph.astype(float)
    if pc.sum() <= 0 or ph.sum() <= 0:
        return 0.0
    pc /= pc.sum(); ph /= ph.sum()
    return float(min(1.0, max(0.0, np.sum(np.minimum(pc, ph)))))


def control_summary(outcome, X_cur_C, X_hist, Y_hist) -> Optional[Dict]:
    """Membership PS + normalized membership-odds weighting -> transported control summary for one
    historical source. Returns {muC, SE, m, r} or None if unusable."""
    X_cur_C = np.asarray(X_cur_C); X_hist = np.asarray(X_hist); Y_hist = np.asarray(Y_hist)
    n_cur = X_cur_C.shape[0]
    Xp = np.vstack([X_cur_C, X_hist])
    G = np.concatenate([np.ones(n_cur), np.zeros(X_hist.shape[0])]).astype(int)
    try:
        lr = LogisticRegression(max_iter=1000, solver="lbfgs").fit(Xp, G)
        e = np.clip(lr.predict_proba(Xp)[:, 1], PS_CLIP[0], PS_CLIP[1])
    except Exception:
        return None
    e_cur, e_hist = e[:n_cur], e[n_cur:]

    # Common support: keep historical controls within the current-control PS range.
    L, U = e_cur.min(), e_cur.max()
    mask = (e_hist >= L) & (e_hist <= U)
    if mask.sum() < MIN_RETAINED:
        return None
    e_h = e_hist[mask]
    Y_h = Y_hist[mask]
    X_h = X_hist[mask]

    # Transport historical controls to the current covariate distribution:
    # weight by the odds of being a current control, then winsorize and normalize.
    omega = e_h / (1.0 - e_h)
    omega = np.minimum(omega, np.percentile(omega, WINSOR))
    omega *= len(omega) / omega.sum()  # mean weight 1

    m = min(kish_ess(omega), int(mask.sum()))
    if m < MIN_ESS:
        return None

    # PS-weighted, covariate-adjusted control summary. Covariates are centered at
    # the current control mean, so the intercept is the control outcome at the
    # current covariate centroid -- the parameter the analysis-stage model places
    # the unit information prior on (continuous: conditional mean at the centroid;
    # binary: conditional log-odds at the centroid). This weighted working
    # regression addresses measured covariate shift under adequate overlap and
    # modeling; it is not a doubly robust estimator.
    Xd = np.column_stack([np.ones(len(Y_h)), X_h - X_cur_C.mean(0)])
    try:
        if outcome == "continuous":
            res = sm.WLS(Y_h, Xd, weights=omega).fit()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(Y_h, Xd, family=sm.families.Binomial(),
                            freq_weights=omega).fit()
        muC, SE = float(res.params[0]), float(res.bse[0])
    except Exception:
        return None
    if not (np.isfinite(muC) and np.isfinite(SE) and SE > 0):
        return None

    r = membership_overlap(e_cur, e_h, omega)
    return {"muC": muC, "SE": SE, "m": float(m), "r": float(r)}


def current_control_summary(outcome, X_cur_C, Y_cur_C) -> Optional[Dict]:
    """Covariate-adjusted intercept of the CURRENT control arm at its own centroid:
    the reference (muC_cur, SE_cur) that the prior-data-conflict discount compares
    each historical source against. Covariates are centered at the current control
    mean, so the intercept is the current control outcome at exactly the centroid
    that control_summary() centers the historical summaries on, making the
    discrepancy a like-for-like control-mean comparison. Uses control data only, so
    the treatment effect theta is never involved. Returns {muC, SE} or None."""
    X = np.asarray(X_cur_C); Y = np.asarray(Y_cur_C)
    n = len(Y)
    Xd = np.column_stack([np.ones(n), X - X.mean(0)])
    try:
        if outcome == "continuous":
            res = sm.OLS(Y, Xd).fit()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(Y, Xd, family=sm.families.Binomial()).fit()
        muC, SE = float(res.params[0]), float(res.bse[0])
    except Exception:
        return None
    if not (np.isfinite(muC) and np.isfinite(SE) and SE > 0):
        return None
    return {"muC": muC, "SE": SE}


def conflict_discount(muC_k, SE_k, muC_cur, SE_cur, c, shape="exp") -> float:
    """Single-prior prior-data-conflict discount rho_k in (0, 1] for source k.

        T_k = (muC_k - muC_cur)^2 / (SE_k^2 + SE_cur^2)
        rho_k = min(1, exp(c - T_k)) = exp(-max(0, T_k - c))

    T_k is a heuristic standardized current-vs-historical control-summary
    discrepancy based on model-based standard errors. The discount equals one until
    T_k exceeds the prespecified threshold c, then decays exponentially in T_k-c.
    It is a deterministic scale on the borrowed information-unit count, not a
    calibrated test statistic or mixture component. Its information contribution is
    bounded because M_k = m_k r_k rho_k <= n_k; this bound does not guarantee type-I
    error control."""
    v = SE_k * SE_k + SE_cur * SE_cur
    if not (np.isfinite(v) and v > 0):
        return 1.0
    T = (muC_k - muC_cur) ** 2 / v
    if not np.isfinite(T) or T <= c:
        return 1.0
    # shape="exp": the default exponential decay exp(c - T);
    # shape="invT": a gentler 1/T decay c / T, both equal to 1 at T = c and decreasing
    # in T. The alternative is used only by the sensitivity analysis.
    if shape == "invT":
        return float(c / T)
    return float(np.exp(c - T))


def raw_control_summary(outcome, X_cur_C, X_hist, Y_hist) -> Optional[Dict]:
    """No-PS ablation: covariate-adjusted but UNWEIGHTED historical-control
    summary, with no transport, no trimming, m = n_k and r = 1. It shares the
    centered covariate adjustment of control_summary so the only difference from
    PS-UIP is the propensity-score stage, which isolates the value of the PS."""
    X_cur_C = np.asarray(X_cur_C)
    Xh = np.asarray(X_hist)
    Y = np.asarray(Y_hist)
    n = len(Y)
    Xd = np.column_stack([np.ones(n), Xh - X_cur_C.mean(0)])
    try:
        if outcome == "continuous":
            res = sm.WLS(Y, Xd, weights=np.ones(n)).fit()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(Y, Xd, family=sm.families.Binomial()).fit()
        muC, SE = float(res.params[0]), float(res.bse[0])
    except Exception:
        return None
    if not (np.isfinite(muC) and np.isfinite(SE) and SE > 0):
        return None
    return {"muC": muC, "SE": SE, "m": float(n), "r": 1.0}
